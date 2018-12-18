import pymongo
from datetime import datetime
from log import log


class Database(object):
    def __init__(self):
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db = self.client['twitter']
        self.tweets = self.db.tweets

    def add_tweets(self, tweets):
        existing = set([
            tweet['id']
            for tweet in self.tweets.find(projection=['id'])
        ])
        new_tweets = list(filter(lambda tweet: tweet['id'] not in existing, tweets))

        if len(new_tweets) > 0:
            self.tweets.insert_many(new_tweets)

    def get_tweets(self, limit=None):
        kwargs = {'limit': limit} if limit else {}
        for tweet in self.tweets.find(**kwargs):
            yield tweet


    def count_tweets(self):
        return self.tweets.find().count()
