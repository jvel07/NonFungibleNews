import feedparser
import pandas as pd
from datetime import datetime


class RSSFeedReader:
    def __init__(self, url="https://rss.app/feeds/t4e3HRo55PJWeWSl.xml"):
        self.url = url
        self.feed = None
        self.df = None

    def fetch(self):
        self.feed = feedparser.parse(self.url)
        articles = [{
            'title': entry.title,
            'link': entry.link,
            'published': datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z'),
            'summary': entry.summary
        } for entry in self.feed.entries]

        self.df = pd.DataFrame(articles)
        return self.df

    def get_latest(self, n=5):
        if self.df is None:
            self.fetch()
        return self.df.head(n)