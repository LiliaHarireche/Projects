"""
Find indexes of pro and anti-stereotypical sentences from en.txt to
use it on fast_align results (.align files)

pro-anti_indexes.pkl in ./translations/_aligns/
"""
import os
import pickle

if __name__ == '__main__':
    PATH_PRO = './data/en_pro.txt'
    PATH_ANTI = './data/en_anti.txt'
    PATH_ALL = './data/en.txt'

    # Lists of sentences
    pro_sents = [line.split("\t")[2] for line in open(PATH_PRO, encoding="utf8")]
    ant_sents = [line.split("\t")[2] for line in open(PATH_ANTI, encoding="utf8")]
    all_sents = [line.split("\t")[2] for line in open(PATH_ALL, encoding="utf8")]

    pro_indexes = []
    anti_indexes = []
    # search indexes of each pro and anti-stereotypical sentences
    for ind, sentence in enumerate(all_sents):
        if sentence in ant_sents:
            anti_indexes.append(ind)

        if sentence in pro_sents:  # not an else : anti and pro have only 3168 sentences at all
            pro_indexes.append(ind)

    # NOTE: there are two more sentences than in originals in each indexes list

    # Save indexes as pickle
    with open('./translations/_aligns/pro-anti_indexes.pkl', "wb") as file:
        pickle.dump([pro_indexes, anti_indexes], file)
