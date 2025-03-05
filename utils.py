import functools
from pathlib import Path

import pandas as pd
import requests
import time
from datetime import datetime

def retry_until_file_exists(wait_time=300):  # 5 minutes default wait time
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except FileNotFoundError as e:
                    file_path = next((arg for arg in args if isinstance(arg, (str, Path))), None)
                    print(f"File {file_path} not found. Attempt {attempt}. Waiting {wait_time/60} minutes for file to arrive...")
                    time.sleep(wait_time)
                    attempt += 1
        return wrapper
    return decorator


def generate_image_sdxl(tweet_text, pipe):
    print(f"Generating image for tweet: {tweet_text}")
    pipe = pipe.to("cuda")
    prompt = f"make an artistic and attractive image that from the following: {tweet_text}"
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]
    image_path = f"data/images/{tweet_text[50:60]}.png"
    image.save(image_path)
    print(f"Image saved at {image_path}")
    return image_path

@retry_until_file_exists(wait_time=300)
def get_news_from_csv(file):
    print("Fetching news from neural files...")
    news_df = pd.read_csv(file)
    if "crypto" in file:
        top_5 = news_df.head(5).copy()
        if top_5['symbol'].str.contains('USD', na=False).any():
            next_row = news_df.iloc[5]
            top_5 = top_5.append(next_row, ignore_index=True)
            top_5 = top_5.drop(top_5[top_5['symbol'].str.contains('USD', na=False)].index)
            selected_columns = ['symbol', 'price_usd', 'market_cap_rank', '24h_volume']
            top_5 = top_5[selected_columns]
            top_5['price_usd'] = top_5['price_usd'].round(2)
        return top_5
    else:
        news_df['used_for_posting'] = False
        news_df = news_df.dropna().reset_index(drop=True)
        return news_df

def format_top5_crypto_llm(data):
    print("Formatting top 5 crypto data...")
    rows = []
    for _, row in data.iterrows():
        formatted_row = (
            f"{row['symbol']}: "
            f"price ${row['price_usd']:.2f}, "
            f"rank #{row['market_cap_rank']}, "
            f"volume {row['24h_volume']}"
        )
        rows.append(formatted_row)
    return "\n".join(rows)
