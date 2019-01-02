import csv
import pandas as pd


df1 = pd.read_csv('tweets.csv', sep='\t')

data = []
with open('sentiment_data_clean.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter='\t')
    next(reader)
    for line in reader:
        data.append([line[1], ','.join(line[2:])])

df2 = pd.DataFrame(data, columns=['ItemID', 'tweet'])
df = pd.concat([df1, df2], copy=False).reset_index(drop=True)
df.drop(['ItemID', 'index'], inplace=True, axis=1)
df.index.name = 'index'
df.to_csv('all_tweets.csv', sep='\t')
