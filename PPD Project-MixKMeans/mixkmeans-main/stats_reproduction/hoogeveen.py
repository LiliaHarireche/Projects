"""
    Reproduction of some results of the article :
    'CQADupStack: A Benchmark Data Set for Community
    Question-Answering Research' - Hoogeveen, Doris and Verspoor,
    Karin M. and Baldwin, Timothy


"""
from time import time

import pandas as pd
from data_processing.utils import SubForumStats


def reproduce_stats(subforum):
    """ Reproduce statistics from article """
    dic = {'# threads': len(subforum.questions)}

    # average number of answers per question
    subforum.count_answers()
    dic['av. # a per q'] = subforum.questions['nb_answers'].mean()

    # average number of words per question
    subforum.count_words_questions()
    dic['av. # w per q'] = subforum.questions['nb_words'].mean()

    # average number of words per thread
    subforum.count_words_threads()
    dic['av. # w per t'] = subforum.questions['nb_words_threads'].mean()

    # percentage of duplicates
    subforum.count_duplicates()
    temp = len(subforum.questions['nb_dups'][subforum.questions['nb_dups'] != 0]) # noqa
    dic['% dups'] = temp*100/len(subforum.questions)

    # average number of duplicates per question
    dic['av. # d per q'] = subforum.questions['nb_dups'].mean()
    return dic


if __name__ == '__main__':
    t = time()
    android = SubForumStats('../data/original_data/android_questions.json',
                            '../data/original_data/android_answers.json')
    android.delete_columns()
    android.pre_processing()
    dic = reproduce_stats(android)
    # init dataframe
    df = pd.DataFrame(dic, index=['android'])
    print('execution android {} s'.format(time()-t))
    del android

    t = time()
    gis = SubForumStats('../data/original_data/gis_questions.json',
                        '../data/original_data/gis_answers.json')
    gis.delete_columns()
    gis.pre_processing()
    dic = reproduce_stats(gis)
    # update df
    temp = pd.Series(dic, name='gis')
    df = df.append(temp)
    print('execution gis {} s'.format(time() - t))
    del gis

    t = time()
    physics = SubForumStats('../data/original_data/physics_questions.json',
                            '../data/original_data/physics_answers.json')
    physics.delete_columns()
    physics.pre_processing()
    dic = reproduce_stats(physics)
    # update df
    temp = pd.Series(dic, name='physics')
    df = df.append(temp)
    print('execution physics {} s'.format(time() - t))
    del physics

    t = time()
    stats = SubForumStats('../data/original_data/stats_questions.json',
                          '../data/original_data/stats_answers.json')
    stats.delete_columns()
    stats.pre_processing()
    dic = reproduce_stats(stats)
    # update df
    temp = pd.Series(dic, name='stats')
    df = df.append(temp)
    print('execution stats {} s'.format(time() - t))
    del stats

    # SAVING
    df.to_csv('statistics_hoogeveen.csv')
