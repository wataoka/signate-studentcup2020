from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import scipy.stats as stats

def clean_description(train_df):

    def remove_verbose(desc):
        sentence = []
        for word in desc.split(' '):
            if len(word) == 0:
                continue
            if len(word) == 1:
                continue
            if len(word) == 2:
                continue
            if word[-1] == ' ':
                word = word[:-1]
            if word[0] == ' ':
                word = word[1:]
            if word[-1] == '.':
                word = word[:-1]
            if word[-1] == ',':
                word = word[:-1]
            if word[-1] == ':':
                word = word[:-1]
            if word[-1] == ';':
                word = word[:-1]
            if word[-1] == 's':
                word = word[:-1]
            if word in ['and', 'the', '']:
                continue
            if '-' in word:
                for w in word.split('-'):
                    sentence.append(w)
                continue
            sentence.append(word)
        return ' '.join(sentence)

    df = pd.DataFrame([])
    df['description'] = train_df['description'].apply(remove_verbose)
    df['jobflag'] = train_df['jobflag']

    overlap_tf = df['description'].duplicated(keep=False)
    overlap_ids = df[overlap_tf].index

    cnt = 0
    min2mem = defaultdict(list)
    for i in overlap_ids:
        for j in overlap_ids:
            if i==j:
                continue
            if j in min2mem.keys():
                continue
            if df[i:i+1]['description'].values[0] == df[j:j+1]['description'].values[0]:
                min2mem[i].append(j)

    overlap_df = pd.DataFrame([])
    for i in min2mem.keys():
        jobflag_list = []
        member_ids = [i, *min2mem[i]]
        for j in member_ids:
            jobflag_list.append(df[j:j+1]['jobflag'].values[0])

        jobflag = stats.mode(jobflag_list)
        jobflag = int(jobflag[0])

        overlap_df = pd.concat([overlap_df, df[i:i+1]], axis=0)

    df = train_df.drop(overlap_ids)
    df = pd.concat([df, overlap_df], axis=0)

    return df