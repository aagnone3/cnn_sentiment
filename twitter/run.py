import pymongo
from log import log
from client import get_api, limit_handled
from database import Database


def is_valid_tweet(tweet):
    return 'id' in tweet and 'retweeted' in tweet and tweet.get('lang', '') == 'en'


def main():
    api = get_api()
    log_interval = 1000
    db_interval = 1000
    db = Database()
    log.info("Starting first generator iteration.")
    while True:
        tweets = []
        for iter_num, tweet in enumerate(api.GetStreamSample()):
            if is_valid_tweet(tweet):
                tweets.append(tweet)

            # periodically log
            if iter_num % log_interval == 0:
                log.info("\tGot {} tweets.".format(len(tweets)))

            # insert the tweets into the database
            if iter_num % db_interval == 0:
                log.info("\tInserting {} tweets into the database.".format(len(tweets)))
                db.add_tweets(tweets)
                tweets = []

        log.info("End of current generator iteration. Got {} tweets.".format(len(tweets)))


if __name__ == '__main__':
    main()
