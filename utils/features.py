import os
import sys
import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

sys.path.append('..')
from constants import REPRESENTATIONS_DIR

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

def word2vec(train_df, test_df, mode='mean'):

    filename = os.path.join(REPRESENTATIONS_DIR, 'word2vec.pkl')
    with open(filename, mode='rb') as f:
        word2vec = pickle.load(f)

    def get_vec_list(desc):
        vec_list = []
        for word in desc.split(' '):
            if len(word) == 0 or word == '.':
                continue
            if word not in word2vec.keys():
                continue
            if word[-1] in ['.', ',', ';', ')']:
                word = word[:-1]
            if len(word) == 0:
                continue
            if word[0] in ['(']:
                word = word[1:]
            vec_list.append(word2vec[word])
        vec_list = np.array(vec_list)
        return vec_list

    train_descriptions = train_df['description'].values.tolist()
    train_data = pd.DataFrame([])
    print('Creating train data...')
    for description in tqdm(train_descriptions):
        vec_list = get_vec_list(description)
        if mode == 'mean':
            vec = vec_list.mean(axis=0)
        else:
            print(f'InvalidMode: {mode} is not supported in word2vec.')
            exit(1)
        vec = pd.DataFrame(vec.reshape(1, len(vec)))
        train_data = pd.concat([train_data, vec], axis=0)

    test_descriptions = test_df['description'].values.tolist()
    test_data = pd.DataFrame([])
    print('Creating test data...')
    for description in tqdm(test_descriptions):
        vec_list = get_vec_list(description)
        if mode == 'mean':
            vec = vec_list.mean(axis=0)
        else:
            print(f'InvalidMode: {mode} is not supported in word2vec.')
            exit(1)
        vec = pd.DataFrame(vec.reshape(1, len(vec)))
        test_data = pd.concat([test_data, vec], axis=0)
    
    return train_data, test_data
