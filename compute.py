#!/usr/bin/env python3

# Please not the code comment is inside Code directory

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import glob
import re
import base64
from textblob import TextBlob
import plotly.express as px
# python -m textblob.download_corpora
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import wordninja
from spellchecker import SpellChecker
from collections import Counter
import nltk
import math
import random
import os
import sys
import yaml
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))
stop_words.add("amp")
pd.options.mode.chained_assignment = None
all_vax = ['covaxin', 'sinopharm', 'sinovac', 'moderna',
           'pfizer', 'biontech', 'oxford', 'astrazeneca', 'sputnik']


def vaccine_data_processing(vaccine,path,filename):
    file_dir = path
    file = filename
    # Make a list of dataframes while adding a stick_ticker column
    # dataframes = [pd.read_csv(file, encoding='latin1').assign(vaccine_file=os.path.basename(file).strip(".csv")) for
    #              file in files]
    df = pd.read_csv(file_dir+file, encoding='latin1',
                     quotechar='"', delimiter=',')
    # data pre-processing and cleaning
    df.drop(columns=['id', 'tweetid', 'guid', 'link', 'source', 'lang',
                     'quoted_text', 'tweet_type', 'in_reply_to_screen_name',
                     'in_reply_to_user_id', 'in_reply_to_status_id', 'retweeted_screen_name',
                     'retweeted_user_id', 'retweeted_status_id', 'user_id',
                     'profile_image_url', 'user_statuses_count', 'user_friends_count',
                     'user_followers_count', 'user_created_at', 'user_bio', 'user_location',
                     'user_verified', 'Unnamed: 29'], inplace=True)
    df = df.drop_duplicates('description')
    df['description'].transform(clean_tweet_text)
    df['date'] = pd.to_datetime(df['pubdate']).dt.date
    # applying the TextBlob API onto our tweet data to perform sentiment analysis
    df['polarity'] = df['description'].apply(
        lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['description'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)
    # polarity values ranging from -1 to 1 are used for sentiment analysis
    # converting our data to 3 classes (negative, neutral, and positive) so that we can visualize it
    criteria = [df['polarity'].between(
        -1, -0.01), df['polarity'].between(-0.01, 0.01), df['polarity'].between(0.01, 1)]
    values = ['negative', 'neutral', 'positive']
    df['sentiment'] = np.select(criteria, values, 0)
    vaccine_df, vaccine_timeline = filter_by_vaccy(df, [vaccine])
    vaccine_df.sort_values(by='polarity', ascending=True)[['description', 'pubdate', 'polarity']].reset_index(
        drop=True).head(n=10).to_json('/data/'+vaccine+r'_10_most_negative_tweets.json')
    vaccine_df.sort_values(by='polarity', ascending=False)[['description', 'pubdate', 'polarity']].reset_index(
        drop=True).head(n=10).to_json('/data/'+vaccine+r'_10_most_positive_tweets.json')
    fig = px.bar(vaccine_timeline, x='date', y='count', color='polarity')
    # fig.show()
    fig.write_image("/data/"+vaccine+"_timeseries_polarity_optimised_dataset.png")
    wordcloud_df = vaccine_df
    wordcloud_df['words'] = vaccine_df.description.apply(lambda x: re.findall(r'\w+', x))
    get_smart_clouds(wordcloud_df).savefig("/data/"+vaccine+"_sentiment_wordclouds_optimised_dataset.png", bbox_inches="tight")

# use regular expressions to strip each tweet of mentions, hashtags, retweet information, and links


def clean_tweet_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = text.lower()
    return text


# Function to filter the data to a single vaccine and plot the timeline Note: a lot of the tweets seem to contain
# hashtags for multiple vaccines even though they are specifically referring to one vaccine-- not very helpful!
def filter_by_vaccy(df, vax):
    df_filt = pd.DataFrame()
    for v in vax:
        df_filt = df_filt.append(
            df[df['description'].str.lower().str.contains(v)])
    other_vax = list(set(all_vax) - set(vax))
    for o in other_vax:
        df_filt = df_filt[~df_filt['description'].str.lower().str.contains(o)]
    #     df_filt = df_filt.drop_duplicates()
    timeline = df_filt.groupby(['date']).agg(np.nanmean).reset_index()
    timeline['count'] = df_filt.groupby(
        ['date']).count().reset_index()['retweet_count']
    timeline = timeline[['date', 'count', 'polarity',
                         'retweet_count', 'favorite_count', 'subjectivity']]
    timeline["polarity"] = timeline["polarity"].astype(float)
    timeline["subjectivity"] = timeline["subjectivity"].astype(float)
    return df_filt, timeline


# Advanced word-cloud (positive, negative and neutral separation)
def flatten_list(l):
    return [x for y in l for x in y]


def is_acceptable(word: str):
    return word not in stop_words and len(word) > 2

# Color coding our wordclouds


def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(0, 100%, {random.randint(25, 75)}%)"


def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl({random.randint(90, 150)}, 100%, 30%)"


def yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(42, 100%, {random.randint(25, 50)}%)"

# Reusable function to generate word clouds


def generate_word_clouds(neg_doc, neu_doc, pos_doc):
    # Display the generated image:
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    wordcloud_neg = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(" ".join(neg_doc))
    axes[0].imshow(wordcloud_neg.recolor(
        color_func=red_color_func, random_state=3), interpolation='bilinear')
    axes[0].set_title("Negative Words")
    axes[0].axis("off")

    wordcloud_neu = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(" ".join(neu_doc))
    axes[1].imshow(wordcloud_neu.recolor(
        color_func=yellow_color_func, random_state=3), interpolation='bilinear')
    axes[1].set_title("Neutral Words")
    axes[1].axis("off")

    wordcloud_pos = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(" ".join(pos_doc))
    axes[2].imshow(wordcloud_pos.recolor(
        color_func=green_color_func, random_state=3), interpolation='bilinear')
    axes[2].set_title("Positive Words")
    axes[2].axis("off")

    plt.tight_layout()
#     plt.show();
    return fig


def get_top_percent_words(doc, percent):
    # Returns a list of "top-n" most frequent words in a list
    top_n = int(percent * len(set(doc)))
    counter = Counter(doc).most_common(top_n)
    top_n_words = [x[0] for x in counter]
    # print(top_n_words)
    return top_n_words


def clean_document(doc):
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()

    # Lemmatize words (needed for calculating frequencies correctly )
    doc = [lemmatizer.lemmatize(x) for x in doc]

    # Get the top 10% of all words. This may include "misspelled" words
    top_n_words = get_top_percent_words(doc, 0.1)

    # Get a list of misspelled words
    misspelled = spell.unknown(doc)

    # Accept the correctly spelled words and top_n words
    clean_words = [x for x in doc if x not in misspelled or x in top_n_words]

    # Try to split the misspelled words to generate good words (ex. "lifeisstrange" -> ["life", "is", "strange"])
    words_to_split = [
        x for x in doc if x in misspelled and x not in top_n_words]
    split_words = flatten_list([wordninja.split(x) for x in words_to_split])

    # Some splits may be nonsensical, so reject them ("llouis" -> ['ll', 'ou', "is"])
    clean_words.extend(spell.known(split_words))

    return clean_words


def get_log_likelihood(doc1, doc2):
    doc1_counts = Counter(doc1)
    doc1_freq = {
        x: doc1_counts[x]/len(doc1)
        for x in doc1_counts
    }

    doc2_counts = Counter(doc2)
    doc2_freq = {
        x: doc2_counts[x]/len(doc2)
        for x in doc2_counts
    }

    doc_ratios = {
        # 1 is added to prevent division by 0
        x: math.log((doc1_freq[x] + 1)/(doc2_freq[x]+1))
        for x in doc1_freq if x in doc2_freq
    }

    top_ratios = Counter(doc_ratios).most_common()
    top_percent = int(0.1 * len(top_ratios))
    return top_ratios[:top_percent]

# Function to generate a document based on likelihood values for words


def get_scaled_list(log_list):
    counts = [int(x[1]*100000) for x in log_list]
    words = [x[0] for x in log_list]
    cloud = []
    for i, word in enumerate(words):
        cloud.extend([word]*counts[i])
    # Shuffle to make it more "real"
    random.shuffle(cloud)
    return cloud


def get_smart_clouds(df):

    neg_doc = flatten_list(df[df['sentiment'] == 'negative']['words'])
    neg_doc = [x for x in neg_doc if is_acceptable(x)]

    pos_doc = flatten_list(df[df['sentiment'] == 'positive']['words'])
    pos_doc = [x for x in pos_doc if is_acceptable(x)]

    neu_doc = flatten_list(df[df['sentiment'] == 'neutral']['words'])
    neu_doc = [x for x in neu_doc if is_acceptable(x)]

    # Clean all the documents
    neg_doc_clean = clean_document(neg_doc)
    neu_doc_clean = clean_document(neu_doc)
    pos_doc_clean = clean_document(pos_doc)

    # Combine classes B and C to compare against A (ex. "positive" vs "non-positive")
    top_neg_words = get_log_likelihood(
        neg_doc_clean, flatten_list([pos_doc_clean, neu_doc_clean]))
    top_neu_words = get_log_likelihood(
        neu_doc_clean, flatten_list([pos_doc_clean, neg_doc_clean]))
    top_pos_words = get_log_likelihood(
        pos_doc_clean, flatten_list([neu_doc_clean, neg_doc_clean]))

    # Generate syntetic a corpus using our loglikelihood values
    neg_doc_final = get_scaled_list(top_neg_words)
    neu_doc_final = get_scaled_list(top_neu_words)
    pos_doc_final = get_scaled_list(top_pos_words)

    # Visualise our synthetic corpus
    fig = generate_word_clouds(neg_doc_final, neu_doc_final, pos_doc_final)
    return fig


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # vaccine = input("Please enter vaccine name for which you want the sentiment analysis computation: \n")
    # vaccine_data_processing(vaccine.lower())

    # Make sure that at least one argument is given, that is either 'encode' or 'decode'
    # if len(sys.argv) != 2:
    #     print(f"Usage: {sys.argv[0]} Please enter vaccine name for which you want the sentiment analysis computation:")
    #     exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    path=sys.argv[2]
    filename=sys.argv[3]
    # print(yaml.dump({command}))
    vaccine_data_processing(os.environ["INPUT"],os.environ["PATH"],os.environ["FILENAME"])

    # Print the result with the YAML package
    print(yaml.dump({ "output": 'Done' }))
