import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import signal
import time

import pytz
import requests
import pandas as pd
from datetime import datetime, timedelta
import tweepy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
from typing import Dict, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_instance import LLMSummarizer
from news_prompts import news_sys_prompt, news_writing_shorten_prompt

class TrendTracker:
    def __init__(self, newsapi_key: str, twitter_bearer_token: str):
        self.newsapi_key = newsapi_key
        self.twitter_bearer_token = twitter_bearer_token
        self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)

        self.summarizer = LLMSummarizer(
            # model_name = "meta-llama/Llama-3.1-8B-Instruct",
            model_name = "meta-llama/Llama-3.1-8B-Instruct",
            system_prompt = news_sys_prompt,
            )

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        self.topic_keywords = {
            'cryptocurrency': ['Bitcoin ETF', 'layer2', 'DeFi', 'zkSync', 'Solana', 'memecoin'],
            'NFT': ['NFT gaming', 'blue chip NFT', 'NFT staking', 'dynamic NFT'],
            'metaverse': ['web3 gaming', 'virtual land', 'metaverse'],
            'memecoin': self.get_top_memecoins(),
            'artificial intelligence': ['AI agents', 'AGI'],
            'crypto ai': ['AI memes', 'AI blockchain', 'decentralized AI']
        }

    def cleanup(self):
        """Release GPU memory"""
        if hasattr(self, 'summarizer'):
            # Move models to CPU
            self.summarizer.pipeline.model.to('cpu')
            torch.cuda.empty_cache()

    def summarize_text(self, text: str) -> str:
        output = self.summarizer.summarize_text(
            text,
            max_new_tokens=256,
            rewrite_prompt=news_writing_shorten_prompt
        )
        summarized_styled = output

        return summarized_styled

    def extract_article_content(self, url: str) -> Dict:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            content_selectors = ['article', '[class*="article"]', '[class*="content"]', '[class*="story"]', 'main']
            content = ""

            for selector in content_selectors:
                if main_content := soup.select_one(selector):
                    paragraphs = main_content.find_all('p')
                    content = ' '.join(p.get_text() for p in paragraphs)
                    if len(content) > 200:
                        break

            content = self.clean_text(content)
            key_insights = self.summarize_text(content[:1000]) if content else ""

            return {
                'content': content[:5000],
                'key_insights': key_insights
            }
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return {'content': '', 'key_insights': ''}

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def calculate_engagement_score(self, article: Dict) -> float:
        base_score = 1.0
        pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
        current_time = datetime.now(pub_date.tzinfo)
        hours_old = (current_time - pub_date).total_seconds() / 3600
        time_factor = np.exp(-hours_old / 24)

        major_sources = ['Bloomberg', 'Reuters', 'CoinDesk', 'The Block', 'Forbes']
        source_factor = 1.5 if article['source'] in major_sources else 0.5

        major_personas = ['Vitalik Buterin', 'Elon Musk', 'Satoshi Nakamoto', 'Mark Cuban',
                          'Michael Saylor', 'CZ Binance', 'Charlie Lee', 'Andreas Antonopoulos',
                          'Sam Altman', 'Tim Draper', 'Andrew Tate', 'Brian Armstrong',
                          'Binance', 'Coinbase', 'Justin Sun', 'BAYC', 'Pudgy Penguins', 'CryptoPunks',
                          ]
        presidents = ['Donald Trump', 'Nayib Bukele']

        source_factor += 0.2 if any(persona.lower() in article['content'].lower() for persona in presidents) else 0.0
        source_factor += 0.5 if any(persona.lower() in article['content'].lower() for persona in major_personas) else 0.0

        black_list = ['ZyCrypto', 'India', 'Tistory.com']
        source_factor = 0.0 if any(source.lower() in article['source'].lower() for source in black_list) else source_factor


        # TODO: increase score if article mentions any of the trending keywords
        source_factor += 0.5 if any(keyword.lower() in article['content'].lower() for keyword in self.get_trending_keywords()) else 0.0

        relevance_factor = 1.0
        for keyword in self.get_trending_keywords():
            if keyword.lower() in article['content'].lower():
                relevance_factor += 0.2

        return base_score * time_factor * source_factor * relevance_factor

    def get_top_memecoins(self) -> List[str]:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/search/trending')
            coins = response.json()['coins']
            memecoins = [coin['item']['symbol'].upper() for coin in coins
                         if any(tag.lower() in ['meme', 'memecoin']
                                for tag in coin['item'].get('tags', []))]
            return memecoins[:5] if memecoins else ['DOGE', 'PEPE', 'BONK', 'WIF']
        except Exception as e:
            print(f"Error fetching memecoins: {str(e)}")
            return ['DOGE', 'PEPE', 'BONK', 'WIF']

    # this basically gets top tokens 24h and sentiment crypto
    def get_trending_keywords(self) -> List[str]:
        try:
            trending_response = requests.get('https://api.coingecko.com/api/v3/search/trending')
            trending_coins = [coin['item']['symbol'].upper() for coin in trending_response.json()['coins']]

            fear_greed_response = requests.get('https://api.alternative.me/fng/')
            sentiment = fear_greed_response.json()['data'][0]['value_classification'].lower()

            return trending_coins + [sentiment]
        except Exception as e:
            print(f"Error fetching trending keywords: {str(e)}")
            return []

    @staticmethod
    def remove_semantic_duplicates(data, similarity_threshold=0.85):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        titles = data['title'].tolist()
        embeddings = model.encode(titles)

        similarity_matrix = cosine_similarity(embeddings)

        indices_to_drop = set()

        for i in range(len(titles)):
            if i in indices_to_drop:
                continue

            for j in range(i + 1, len(titles)):
                if j in indices_to_drop:
                    continue

                if similarity_matrix[i][j] > similarity_threshold:
                    # keep the earlier one (i) and mark the later one (j) for dropping
                    indices_to_drop.add(j)

        # keep only the rows that aren't marked for dropping
        return data.drop(index=data.index[list(indices_to_drop)])

    def get_news(self, topic: str, days: int = 1) -> List[Dict]:
        # TODO: add these kind of things like Trump changing his mind on crypto 3 years ago https://x.com/cryptobeastreal/status/1882461437031059691
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': topic,
            'from': from_date,
            'sortBy': 'popularity',
            'language': 'en',
            'apiKey': self.newsapi_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json()['articles']

            detailed_articles = []
            for article in articles[:5]:
                article_details = self.extract_article_content(article['url'])
                article_data = {
                    'title': article['title'],
                    'source': article['source']['name'],
                    'url': article['url'],
                    'published_at': article['publishedAt'],
                    'topic': topic,
                    'content': article_details['content'],
                    'key_insights': article_details['key_insights']
                }
                article_data['engagement_score'] = self.calculate_engagement_score(article_data)
                detailed_articles.append(article_data)

            return sorted(detailed_articles, key=lambda x: x['engagement_score'], reverse=True)
        except Exception as e:
            print(f"Error fetching news for {topic}: {str(e)}")
            return []

    def get_coingecko_trends(self) -> List[Dict]:
        try:
            # TODO: https://www.binance.com/en/support/announcement/new-cryptocurrency-listing?c=48&navId=48
            response = requests.get('https://api.coingecko.com/api/v3/search/trending')
            response.raise_for_status()

            coins = response.json()['coins']
            trends = []

            for coin in coins:
                item = coin['item']
                trend = {
                    'name': item['name'],
                    'symbol': item['symbol'].upper(),
                    'market_cap_rank': item['market_cap_rank'],
                    'price_btc': item['price_btc'],
                    'price_usd': item['data']['price'],
                    '24h_volume': item.get('data', {}).get('total_volume', 0),
                    'market_cap': item.get('data', {}).get('market_cap', 0),
                    'price_change_24h': item.get('data', {}).get('price_change_percentage_24h', 0)
                }
                trends.append(trend)

            return sorted(trends, key=lambda x: x['market_cap_rank'] if x['market_cap_rank'] else float('inf'))
        except Exception as e:
            print(f"Error fetching CoinGecko trends: {str(e)}")
            return []

    def get_twitter_trends(self, topic: str) -> List[Dict]:
        try:
            tweets = self.twitter_client.search_recent_tweets(
                query=f"{topic} -is:retweet lang:en",
                tweet_fields=['created_at', 'public_metrics'],
                max_results=10
            )

            if not tweets.data:
                return []

            processed_tweets = []
            for tweet in tweets.data[:5]:
                tweet_text = tweet.text
                summary = self.summarize_text(tweet_text)

                engagement_rate = (tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count']) / 100
                trending_matches = [keyword for keyword in self.get_trending_keywords()
                                    if keyword.lower() in tweet_text.lower()]

                tweet_data = {
                    'text': tweet_text,
                    'summary': summary,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'topic': topic,
                    'engagement_rate': engagement_rate,
                    'trending_keywords': trending_matches
                }
                processed_tweets.append(tweet_data)

            return sorted(processed_tweets, key=lambda x: x['engagement_rate'], reverse=True)
        except Exception as e:
            print(f"Error fetching Twitter trends for {topic}: {str(e)}")
            return []

def retrieve_news(est_time_str):
    load_dotenv('.env')
    try:
        tracker = TrendTracker(
            newsapi_key=os.getenv('NEWS_API_KEY_2'),
            twitter_bearer_token=os.getenv('TWITTER_BEARER_TOKEN')
        )

        all_data = {
            'news': [],
            'tweets': [],
            'crypto': tracker.get_coingecko_trends()
        }

        hour = est_time_str[-2:]
        print(f"Retrieving news at {hour} EST...\n")
        for topic, keywords in tracker.topic_keywords.items():
            print(f"Processing {topic}...")
            for keyword in keywords:
                news = tracker.get_news(f"{topic} {keyword}")
                all_data['news'].extend(news)

        for data_type, data in all_data.items():
            df = pd.DataFrame(data)
            if data_type == 'news':
                df = tracker.remove_semantic_duplicates(df).reset_index(drop=True)
            file_name = f'trending_{data_type}_{est_time_str}.csv'
            df.to_csv(file_name, index=False)
            print(f"Saved {len(df)} unique {data_type} to {file_name}")

        print("\nProcessing complete. Check CSV files for detailed results.")
        return all_data
    finally:
        # Clean up GPU memory when done
        tracker.cleanup()
        torch.cuda.empty_cache()

def get_est_hour():
    utc_now = datetime.now(pytz.utc)
    est_now = utc_now.astimezone(pytz.timezone('US/Eastern'))
    return est_now

def get_local_budapest_hour():
    utc_now = datetime.now(pytz.utc)
    local_now = utc_now.astimezone(pytz.timezone('Europe/Budapest'))
    return local_now

def signal_handler(signum, frame):
    global running
    print("\nShutdown signal received. Waiting for current cycle to complete...")
    running = False


# retrieve_news_est_times = [2, 15, 21]
def run_continuous_tracker(retrieve_news_est_times):
    """
    Run the trend tracker at specific hours with proper error handling and shutdown capabilities.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_dotenv('.env')

    print("Starting trend tracking...")
    print("Press CTRL+C to stop safely")

    # track of which hours we've already checked today
    hours_checked = {hour: False for hour in retrieve_news_est_times}
    last_check_date = None

    while running:
        try:
            # est_time = get_est_hour()
            est_time = get_local_budapest_hour()
            current_date = est_time.strftime('%Y-%m-%d')
            current_hour = int(est_time.hour)

            # reset checks on a new day
            if current_date != last_check_date:
                hours_checked = {hour: False for hour in retrieve_news_est_times}
                last_check_date = current_date

            # check if running for current hour
            if current_hour in retrieve_news_est_times and not hours_checked[current_hour]:
                print(f"Time to retrieve news, it's {current_hour}:00!")
                est_time_str = est_time.strftime('%Y-%m-%d %H')
                retrieve_news(est_time_str)
                hours_checked[current_hour] = True
            else:
                print(f"Skipping news retrieval at {current_hour}:00 EST. Next retrieval times: ",
                      [h for h, checked in hours_checked.items() if not checked])

            for _ in range(900):  # 300 seconds = 5 minutes
                if not running:
                    break
                time.sleep(1)

        except Exception as e:
            print(f"Error during tracking cycle: {str(e)}")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)

        if not running:
            print("Shutting down gracefully...")
            break


running = True

if __name__ == "__main__":
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # all_news = retrieve_news(try_now=True)
    retrieve_news_est_times = [12]
    run_continuous_tracker(retrieve_news_est_times=retrieve_news_est_times)