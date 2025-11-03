#!/usr/bin/python

import os
import csv
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


def print_wordcloud(path):
    df = pd.read_csv(csv_path).sample(n=5000).reset_index(drop=True)
    text = ""
    for row in df["review"]:
        text = text + " " + row

    wc = WordCloud(max_font_size=50, max_words=100,
                   background_color='white').generate(text)

    plt.figure('Word Cloud of reviews')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def print_histogram(path):
    df = pd.read_csv(csv_path).sample(n=25000).reset_index(drop=True)

    plt.hist(df['rating_from_one_to_ten'], color='blue', edgecolor='black',
             bins=int(180 / 5))

    # seaborn histogram
    sns.distplot(df['rating_from_one_to_ten'], hist=True, kde=False,
                 bins=int(180 / 5), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Histogram of Review points')
    plt.xlabel('Review points')
    plt.ylabel('Number of occurences')
    plt.show()


def print_confusion_matrix(y_true, y_pred, label=''):
    # Thanks stackoverflow https://stackoverflow.com/a/50326049
    a = confusion_matrix(y_true, y_pred)
    b = index = [
        'true:1',
        'true:2',
        'true:3',
        'true:4',
        'true:7',
        'true:8',
        'true:9',
        'true:10'
    ]
    c = columns = [
        'pred:1',
        'pred:2',
        'pred:3',
        'pred:4',
        'pred:7',
        'pred:8',
        'pred:9',
        'pred:10'
    ]
    cmtx = pd.DataFrame(a, b, c)
    print('# %s%sonfusion matrix' % (label, ' c'if label else 'C'))
    print(cmtx, '\n')


def cleanup_text(df):
    corpus = []

    # See https://www.nltk.org/book/ch02.html 4.1   Wordlist Corpora
    nltk.download('stopwords')

    for row in range(0, df.shape[0]):
        print("Status: ", (row / df.shape[0] * 100), "% done!")
        # First replace punctuation and other non letters with space
        # by matching strings that contain a non-letter
        rev = df['review'][row]
        review = re.sub('[^a-zA-Z]', ' ', df['review'][row])

        review = review.lower()
        review = review.split()

        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if word not in set(
            stopwords.words('english'))]

        review = ' '.join(review)

        corpus.append(review)

    return corpus


def process_data(directory_pos, directory_neg, csv_path):
    csv_f = open(csv_path, 'w', encoding='UTF8', newline='')
    csv_writer = csv.writer(csv_f)
    header = ["review", "rating_from_one_to_ten"]
    csv_writer.writerow(header)
    for filename in os.listdir(directory_neg):
        file_path = os.path.join(directory_neg, filename)
        # extract the rating of the filename
        rating = filename.split("_")[1].split(".")[0]
        with open(file_path, encoding="UTF8") as file:
            lines = [line.rstrip() for line in file]
        lines = "".join(lines).replace(",", "")
        row = [lines, rating]
        csv_writer.writerow(row)

    for filename in os.listdir(directory_pos):
        file_path = os.path.join(directory_pos, filename)
        # extract the rating of the filename
        rating = filename.split("_")[1].split(".")[0]
        with open(file_path, encoding="UTF8") as file:
            lines = [line.rstrip() for line in file]
        lines = "".join(lines).replace(",", "")
        row = [lines, rating]
        csv_writer.writerow(row)


if __name__ == '__main__':
    # assign directory
    directory_neg = 'C:\\Users\\achil\\Google Drive\\TUM\\Semester_7_Erasmus\\Machine_Learning\\Übungen\\ics0030-machine-learning\\Lab 5\\Data\\IMDB_DATASET\\aclImdb\\train\\neg'
    directory_pos = 'C:\\Users\\achil\\Google Drive\\TUM\\Semester_7_Erasmus\\Machine_Learning\\Übungen\\ics0030-machine-learning\\Lab 5\\Data\\IMDB_DATASET\\aclImdb\\train\\pos'

    csv_path = 'C:\\Users\\achil\\Google Drive\\TUM\\Semester_7_Erasmus\\Machine_Learning\\Übungen\\ics0030-machine-learning\\Lab 5\\Data\\ratings.csv'
    csv_f = open(csv_path, 'r', encoding='UTF8', newline='')
    csv_writer = csv.writer(csv_f)

    #process_data(directory_pos, directory_neg, csv_path)

    # print_wordcloud(csv_path)
    print_histogram(csv_path)

    df = pd.read_csv(csv_path)
    df = df.sample(n=25000).reset_index(drop=True)

    corpus = cleanup_text(df)

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.loc[:, 'rating_from_one_to_ten'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    print_confusion_matrix(y_test, clf.predict(X_test))

exit("job finished ;)")
