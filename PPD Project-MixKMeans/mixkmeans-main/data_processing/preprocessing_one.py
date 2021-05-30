"""
Preprocessing execution of both questions and answers datasets
"""
import pickle
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse

from data_processing.utils import SubForum, get_all_words, reduce_words

if __name__ == '__main__':
    os.chdir('../../')
    # ----------------------
    # Preprocessing
    # ----------------------

    android = SubForum('./data/original_data/android_questions.json',
                       './data/original_data/android_answers.json')
    # only android for the moment, needs to do that to deal with memory
    android.change_ids('a')
    android.pre_processing()

    gis = SubForum('./data/original_data/gis_questions.json',
                   './data/original_data/gis_answers.json')
    gis.change_ids('g')
    gis.pre_processing()
    android += gis
    del gis

    physics = SubForum('./data/original_data/physics_questions.json',
                       './data/original_data/physics_answers.json')
    physics.change_ids('p')
    physics.pre_processing()
    android += physics
    del physics

    stats = SubForum('./data/original_data/stats_questions.json',
                     './data/original_data/stats_answers.json')
    stats.change_ids('s')
    stats.pre_processing()
    android += stats
    del stats

    # ----------------------
    # Vocabulary
    # ----------------------

    vocab = reduce_words(get_all_words(android))  # android contains all the 4 datasets

    with open('./data/data_preprocess/vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)

    # ----------------------
    # Documents-Terms-Matrix
    # ----------------------

    # Occurences
    vectorizer = CountVectorizer(vocabulary=vocab)
    Q_count = vectorizer.fit_transform(list(android.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
    A_count = vectorizer.fit_transform(list(android.answers['body'].apply(lambda sentence: ' '.join(sentence)))) # noqa
    sparse.save_npz('./data/data_preprocess/dtm_questions_occ.npz', Q_count)
    sparse.save_npz('./data/data_preprocess/dtm_answers_occ.npz', A_count)

    android.remove_body()
    android.questions.to_csv('./data/data_preprocess/subforum_questions.csv')
    android.answers.to_csv('./data/data_preprocess/subforum_answers.csv')

    del android

    # TF IDF
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    Q_tfidf = transformer.fit_transform(Q_count)
    A_tfidf = transformer.fit_transform(A_count)
    sparse.save_npz('./data/data_preprocess/dtm_questions_tfidf.npz', Q_tfidf)
    sparse.save_npz('./data/data_preprocess/dtm_answers_tfidf.npz', A_tfidf)
