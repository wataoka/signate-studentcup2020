import os
import sys
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

def get_frequent_words(df, threshold=100):

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

def get_important_words(train_df, min_num_word=10, th_score=0.4):

    # description
    train_descriptions = train_df['description'].values.tolist()

    # count num_appear for each classes
    num_appear = {}
    for i, sentence in enumerate(train_descriptions):
        jobflag = int(train_df[i:i+1]['jobflag'].values)

        words = sentence.split(' ')
        for word in words:

            if len(word) == 0 or word == '.':
                continue
            if word[-1] == '.' or word[-1] == ',':
                word = word[:-1]
            
            if not word in num_appear.keys():
                num_appear[word] = [0, 0, 0, 0]
            num_appear[word][jobflag-1] += 1
        
    # scoring
    scores = {}
    for word, appears in num_appear.items():
        if sum(appears) <= min_num_word:
            continue
        score = 0
        for appear in appears:
            score = max(score, appear/sum(appears))
        scores[word] = score
    
    # select important words
    scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    important_words = []
    for word, score in scores:
        if score < th_score:
            break
        important_words.append(word)

    return important_words

def importance(train_df, test_df):

    def count(desc, target):
        ans = 0
        for word in desc.split(' '):
            if len(word) == 0 or word == '.':
                continue
            if word[-1] == '.' or word[-1] == ',':
                word = word[:-1]
            if word == target:
                ans += 1
        return ans
    
    important_words = get_important_words(train_df)

    train_data = pd.DataFrame([])
    print('Creating train data...')
    for word in tqdm(important_words):
        train_data[word] = train_df['description'].apply(count, target=word)

    test_data = pd.DataFrame([])
    print('Creating test data...')
    for word in tqdm(important_words):
        test_data[word] = test_df['description'].apply(count, target=word)
    
    return train_data, test_data