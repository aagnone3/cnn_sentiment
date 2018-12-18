import csv
import json

from database import Database

def main():
    db = Database()
    tweets = [
        tweet['text']
        for tweet in db.get_tweets()
    ]

    with open('tweets.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['index', 'tweet'])
        for i, tweet in enumerate(tweets):
            writer.writerow([i, tweet.replace('\n', '').replace('\r', '')])

    # with open('tweets.json', 'w') as fp:
    #     json.dump({i: tweet for i, tweet in enumerate(tweets)}, fp, indent=4)


if __name__ == '__main__':
    main()
