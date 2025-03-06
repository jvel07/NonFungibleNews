import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import re
import time
from datetime import datetime, timedelta

import backoff
import requests
import schedule
import torch
import tweepy
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

from diffusers import StableDiffusion3Pipeline
from llm_instance import LLMSummarizer
from news_prompts import news_writing_shorten_prompt
from utils import generate_image_sdxl, get_news_from_csv


@dataclass
class TwitterConfig:
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str

    def create_client(self) -> Tuple[tweepy.Client, tweepy.API]:
        client = tweepy.Client(
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )

        auth = tweepy.OAuth1UserHandler(
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )
        api = tweepy.API(auth, timeout=(160,600), wait_on_rate_limit=True, retry_count=8, retry_delay=60)

        return client, api


@dataclass
class SchedulerConfig:
    posts_per_day: int = 10
    # news_retrieval_times: List[str] = field(default_factory=lambda: ["03", "21"])
    model_path: str = "stabilityai/stable-diffusion-3.5-large"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    signature: str = "\n\nAlice K.\nSynthetic Journalist | NFN"
    cache_dir: str = "/srv/data/fuser/hf_cache"
    env_path: str = ".env"
    tweet_history_path: Path = Path("tweet_history.txt")
    include_crypto: bool = False


class NewsContent:
    def __init__(self, timestamp: str, retrieve_time: List[str]):
        # self.news_path = f'trending_news_{timestamp} {retrieve_time[0]}.csv'
        self.news_path = f'trending_news_{timestamp} {retrieve_time[0]}.csv'
        # self.crypto_path = f'trending_crypto_{timestamp} {retrieve_time[0]}.csv'
        self.crypto_path = f'trending_crypto_{timestamp} {retrieve_time[0]}.csv'
        self._news_content = None
        self._crypto_content = None
        self.reset_needed = False
        self.min_reputation = 0.25

    @property
    def news(self):
        print(f"News path: {self.news_path}")
        if self._news_content is None:  # Only load if not cached
            df = get_news_from_csv(self.news_path)
            # print(f"News content:\n {self._news_content}")
            df = df[df['engagement_score'] >= self.min_reputation].copy().reset_index(drop=True)
            self._news_content = df.sort_values(by='engagement_score', ascending=False).head(12).reset_index(drop=True)
            print(f"News content after filtering (engagement >= {self.min_reputation}):\n {self._news_content}")
        return self._news_content

    @property
    def crypto(self):
        if self._crypto_content is None and self.crypto_path:  # Only load if not cached
            self._crypto_content = get_news_from_csv(self.crypto_path)
        return self._crypto_content

    def reset_usage_flags(self):
        """Reset 'used_for_posting' flags when all content has been used"""
        if self._news_content is not None:
            self._news_content['used_for_posting'] = False
        if self._crypto_content is not None:
            self._crypto_content['used_for_posting'] = False
        self.reset_needed = False


class TwitterPoster:
    def __init__(self, twitter_config: TwitterConfig):
        self.client, self.api = twitter_config.create_client()
        self.history_path = Path("tweet_history.txt")

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(requests.exceptions.ConnectionError,
                   requests.exceptions.HTTPError),
        max_tries=10,
        max_value=60
    )
    def post_tweet(self, news_content: str, crypto_content: Optional[str], image_path: Path,
                   include_crypto: bool = False) -> bool:
        print("Getting stuff ready to post tweet...")
        try:
            news_content = news_content.replace('"', '')
            content_options = [news_content]

            if include_crypto and crypto_content:
                content_options.append(crypto_content)

            tweet = random.choice(content_options)
            print(f"tweet:\n {tweet}\n")

            self._save_tweet_history(tweet)

            media = self.api.media_upload(str(image_path), chunked=True)
            self.client.create_tweet(text=tweet, media_ids=[media.media_id])
            print(f"Posted tweet at {datetime.now()}: {tweet}")
            return True

        except Exception as e:
            print(f"Error posting tweet: {e}\n")
            print(f"Number of characters in tweet: {len(tweet)}")
            return False

    def _save_tweet_history(self, tweet: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {tweet}\n{'-' * 80}\n")

# class ImageGenerator:
#     def __init__(self, model_path: str):
#         self.pipe = StableDiffusion3Pipeline.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16
#         )
#
#     def generate(self, prompt: str) -> Path:
#         return generate_image_sdxl(prompt, self.pipe)

class ImageGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        print("Loading Stable Diffusion model...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def cleanup(self):
        if self.pipe is not None:
            self.pipe = self.pipe.to("cpu")
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

    def generate(self, prompt: str) -> Path:
        return generate_image_sdxl(prompt, self.pipe)


class ContentManager:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.shortener = LLMSummarizer(config.llm_model)
        self.image_gen = ImageGenerator(config.model_path)

    def cleanup(self):
        self.shortener.cleanup()
        self.image_gen.cleanup()
        torch.cuda.empty_cache()

    def process_content(self, news_content: str) -> Tuple[str, Path]:
        image_path = self.image_gen.generate(news_content)

        # shorten if needed (280 - 35 for signature and buffer)
        if len(news_content) > 290:
            print(f"Shortening news content. \n Original: {news_content}")
            news_content = self.shortener.rewrite_if_exceeds(
                news_content,
                max_new_tokens=128,
                rewrite_prompt=news_writing_shorten_prompt
            )
        news_content = re.sub(r'(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<![A-Z])\.\s+',
                             '.\n\n',
                             news_content)
        final_tweet = news_content + self.config.signature
        print(f"Final tweet length:\n{len(final_tweet)}")
        return final_tweet, image_path


class TweetScheduler:
    def __init__(self,
                 config: SchedulerConfig,
                 twitter_poster: TwitterPoster,
                 retrieval_times: List[str] = None
                 ):
        self.config = config
        self.content_manager = ContentManager(config)
        self.twitter_poster = twitter_poster
        self.retrieval_times = retrieval_times

    def generate_posting_times(self) -> List[str]:
        # times = [
        #     f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
        #     for _ in range(self.config.posts_per_day)
        # ]
        times = ['03:45', '04:38', '13:54', '15:06', '17:23', '19:49', '21:35', '23:58', '01:45']
        # times = []
        # Add immediate start time
        start_time = (datetime.now() + timedelta(minutes=20)).strftime("%H:%M")
        times.append(start_time)
        return sorted(times)

    def update_used_content(self, news_df):
        """Update and get fresh content from the news DataFrame."""
        # Get a random index from rows where used_for_posting is False
        unused_rows = news_df[news_df['used_for_posting'] == False]

        # If no unused rows, reset all to unused
        # if unused_rows.empty:
        #     news_df['used_for_posting'] = False
        #     unused_rows = news_df

        # Pick a random unused row
        picked_row = unused_rows.sample(1)
        picked_news_index = picked_row.index[0]

        # Mark as used and get content
        news_df.loc[picked_news_index, 'used_for_posting'] = True
        news_content = news_df.loc[picked_news_index, 'key_insights']

        return news_df, news_content

    def schedule_tweets(self) -> List[Tuple[str, str]]:
        """
        Schedule tweets for posting throughout the day.
        Returns:
            List[Tuple[str, str]]: List of scheduled times and content
        """
        timestamp = datetime.now().strftime('%Y-%m-%d')

        # Fixed news retrieval times
        # retrieve_time = ["10"]
        retrieve_time = self.retrieval_times
        # retrieve_time = ["05", "21"]

        try:
            news_content = NewsContent(timestamp, retrieve_time)
            scheduled_content = []  # Initialize list to store scheduled content
            posting_times = self.generate_posting_times()

            print(f"Generated {len(posting_times)} posting times")
            print("Posting times:", posting_times)

            for posting_time in posting_times:
                try:
                    # Update and get fresh content
                    current_news, news_text = self.update_used_content(news_content.news)
                    news_content._news_content = current_news  # Update the cache
                    print(f"Current news:\n{current_news}")

                    # Process content
                    final_content, image_path = self.content_manager.process_content(news_text)

                    # Schedule tweet
                    self._schedule_tweet(posting_time, final_content, news_content.crypto, image_path)

                    # Add to scheduled content list
                    scheduled_content.append((posting_time, final_content))

                    print(f"Successfully scheduled tweet for {posting_time}")
                    print(f"{'=' * 90}\n")

                except Exception as e:
                    print(f"Error scheduling tweet for {posting_time}: {str(e)}")
                    print(f"Stack trace: ", e.__traceback__)
                    continue

            return scheduled_content

        except Exception as e:
            print(f"Fatal error in schedule_tweets: {str(e)}")
            raise
        finally:
            self.content_manager.cleanup()

    def _schedule_tweet(self, time: str, content: str, crypto_content: str, image_path: Path):
        def tweet_job():
            return self.twitter_poster.post_tweet(
                news_content=content,
                crypto_content=crypto_content,
                image_path=image_path,
                include_crypto=self.config.include_crypto
            )

        schedule.every().day.at(time).do(tweet_job)


def main():
    # Load environment variables
    load_dotenv('.env')

    # Initialize configurations
    twitter_config = TwitterConfig(
        api_key=os.getenv('TWITTER_API_KEY'),
        api_secret=os.getenv('TWITTER_API_SECRET_KEY'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    )

    scheduler_config = SchedulerConfig(
        cache_dir="/srv/data/fuser/hf_cache",
        include_crypto=False
    )

    # Initialize components
    retrieve_times = ["11"]
    twitter_poster = TwitterPoster(twitter_config)
    scheduler = TweetScheduler(scheduler_config, twitter_poster, retrieve_times)

    def schedule_daily_tweets():
        print(f"\nScheduling tweets at {datetime.now()}")
        schedule.clear()
        scheduled_tweets = scheduler.schedule_tweets()

        if scheduled_tweets:
            print(f"\nSuccessfully scheduled {len(scheduled_tweets)} tweets:")
            for time, content in scheduled_tweets:
                print(f"- {time}: {content[:100]}...")
        else:
            print("No tweets were scheduled.")
        schedule.every().day.at(f'{retrieve_times[0]}:00').do(schedule_daily_tweets)

    # Initial scheduling
    schedule_daily_tweets()

    print("\nBot is running. Press Ctrl+C to exit.")
    print("Next tweet scheduling at: 03:00")

    try:
        while True:
            now = datetime.now()
            next_run = schedule.next_run()
            if next_run:
                time_until = next_run - now
                print(f"\rNext action in: {time_until}", end="", flush=True)
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot stopped due to error: {e}")
        raise

if __name__ == "__main__":
    main()