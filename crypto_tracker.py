import requests
import time
import tweepy
from datetime import datetime
from dotenv import load_dotenv
import os

# load_dotenv()
load_dotenv('../synthetic_minds/.env')

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')


def setup_twitter():
    """Initialize Twitter API client"""
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)


def send_tweet(twitter_api, message):
    """Send tweet with error handling"""
    try:
        twitter_api.update_status(message)
        print(f"Tweet sent: {message}")
    except Exception as e:
        print(f"Error sending tweet: {e}")


def check_price_alerts(coin_data, twitter_api):
    alerts = []

    for coin, data in coin_data.items():
        day_change = data.get('usd_24h_change', 0)
        hour_change = data.get('usd_1h_change', 0)
        current_price = data.get('usd', 0)

        # 24h alerts (±10%)
        if abs(day_change) >= 10:
            direction = "up" if day_change > 0 else "down"
            alert_msg = f"🚨 {coin.upper()} is {direction} {abs(day_change):.2f}% in the last 24 hours! Price: ${current_price:,.2f} #crypto #{coin}"
            alerts.append(alert_msg)
            send_tweet(twitter_api, alert_msg)

        # 1h alerts (±5%)
        if abs(hour_change) >= 5:
            direction = "up" if hour_change > 0 else "down"
            alert_msg = f"⚡ {coin.upper()} is {direction} {abs(hour_change):.2f}% in the last hour! Price: ${current_price:,.2f} #crypto #{coin}"
            alerts.append(alert_msg)
            send_tweet(twitter_api, alert_msg)

    return alerts


def get_crypto_prices(twitter_api):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum,solana",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_1hr_change": "true"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        # Check for significant price movements
        alerts = check_price_alerts(data, twitter_api)

        # Print current prices and changes
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}]")

        for coin in ['bitcoin', 'ethereum', 'solana']:
            price = data[coin]['usd']
            day_change = data[coin]['usd_24h_change']
            hour_change = data[coin].get('usd_1h_change', 0)

            print(f"{coin.title():8} : ${price:,.2f}  "
                  f"24h: {day_change:+.2f}%  "
                  f"1h: {hour_change:+.2f}%")

        # Print any alerts
        if alerts:
            print("\n🚨 ALERTS TWEETED:")
            for alert in alerts:
                print(f"➡️ {alert}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except KeyError as e:
        print(f"Error parsing data: {e}")


def main():
    print("Crypto Price Tracker with Twitter Alerts Started...")
    print("Monitoring for:")
    print("- ±10% price changes in 24 hours")
    print("- ±5% price changes in 1 hour")
    print("\nInitializing Twitter API...")

    try:
        twitter_api = setup_twitter()
        print("Twitter API initialized successfully!")
        print("\nPress Ctrl+C to stop")

        while True:
            get_crypto_prices(twitter_api)
            time.sleep(60*10)  # Update every minute

    except KeyboardInterrupt:
        print("\nTracker stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()