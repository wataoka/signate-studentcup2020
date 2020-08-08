import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

sys.path.append('..')
from constants import BASE_DIR, DATASET_DIR

def load_dataset():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    submit_sample = pd.read_csv(os.path.join(DATASET_DIR, 'submit_sample.csv'))
    return train, test, submit_sample

def tfidf(train_df, test_df, max_features=None):

    # prepare
    all_df = pd.concat([train_df, test_df], axis=0, sort=True)
    train_description = train_df['description'].values.tolist()
    test_description = test_df['description'].values.tolist()
    all_description = all_df['description'].values.tolist()

    # vectorize
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(all_description)

    train_tfidf = vectorizer.transform(train_description)
    test_tfidf = vectorizer.transform(test_description)
    
    # dataframe
    train_tfidf_df = pd.DataFrame(train_tfidf.toarray())
    test_tfidf_df = pd.DataFrame(test_tfidf.toarray())

    return train_tfidf_df, test_tfidf_df

def frequent_words(df, threshold=100):

    descriptions = df['description'].values.tolist()

    candidates = []
    num_appear = defaultdict(int)
    for sentence in descriptions:
        words = sentence.split(' ')
        for word in words:

            if len(word) == 0:
                continue
            if word == '.':
                continue
            if word[-1] == '.':
                word = word[:-1]

            num_appear[word] += 1
            if num_appear[word] == threshold:
                candidates.append(word)
    
    return candidates