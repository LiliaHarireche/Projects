"""
    Construction of the test set for the evaluating of trained MixKMeans model
"""
import os
from random import sample
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse
import pandas as pd

from data_processing.utils import SubForum


class SubForumTestSet(SubForum):
    def __init__(self, questions_json, answers_json):
        super().__init__(questions_json, answers_json)

    def delete_rows(self):
        """"""
        self.answers = self.answers[self.answers['score'].isin([3, 4, 5])]
        self.questions = self.questions[self.questions.index.isin(self.answers['parentid'])]
        self.answers = self.answers[self.answers['parentid'].isin(self.questions.index)]


if __name__ == '__main__':
    os.chdir('../../')
    # ----------------------
    # Preprocessing
    # ----------------------

    android = SubForumTestSet('./data/original_data/android_questions.json',
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

    physics = SubForumTestSet('./data/original_data/physics_questions.json',
                       './data/original_data/physics_answers.json')
    physics.change_ids('p')
    physics.pre_processing()
    android += physics
    del physics

    stats = SubForumTestSet('./data/original_data/stats_questions.json',
                     './data/original_data/stats_answers.json')
    stats.change_ids('s')
    stats.pre_processing()
    android += stats
    del stats

    # REDUCE DATASET
    id_sample = sample(list(android.answers.index), 2900)
    android.answers = android.answers[android.answers.index.isin(id_sample)]
    android.questions = android.questions[android.questions.index.isin(list(android.answers['parentid']))]

    # get vocab
    with open('./data/data_preprocess/vocab.pkl', 'rb') as file:
        vocab = pickle.load(file)

    # CREATE DOC TERMS MATRIX
    vectorizer = CountVectorizer(vocabulary=vocab)
    Q_count = vectorizer.fit_transform(list(android.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
    A_count = vectorizer.fit_transform(list(android.answers['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa

    android.remove_body()

    # TF IDF
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    Q_tfidf = transformer.fit_transform(Q_count)
    A_tfidf = transformer.fit_transform(A_count)

    # CONSTRUCT FINAL MATRIX
    # processing_two-like

    answers = android.answers
    questions = android.questions

    dtm_a = A_count
    dtm_q = Q_count

    df_dtm_a = pd.DataFrame.sparse.from_spmatrix(dtm_a)
    df_dtm_q = pd.DataFrame.sparse.from_spmatrix(dtm_q)

    df_dtm_a['a_index'] = answers.index  # useless
    df_dtm_a['q_index'] = list(answers['parentid'])
    df_dtm_q['q_index'] = questions.index

    df_dtm_q = df_dtm_q.rename(columns=lambda no: 'q{}'.format(no) if isinstance(no, int) else no)
    df = df_dtm_q.merge(df_dtm_a, right_on='q_index', left_on='q_index')
    indexes_occ = df[['q_index', 'a_index']]
    df = df.drop(['q_index', 'a_index'], axis=1)

    dtm = sparse.csr_matrix(df.values)
    sparse.save_npz('./study/data_test/dtm_occ_test.npz', dtm)

    # tf_idf matrix

    dtm_a = A_tfidf
    dtm_q = Q_tfidf

    df_dtm_a = pd.DataFrame.sparse.from_spmatrix(dtm_a)
    df_dtm_q = pd.DataFrame.sparse.from_spmatrix(dtm_q)

    df_dtm_a['a_index'] = answers.index  # useless
    df_dtm_a['q_index'] = list(answers['parentid'])
    df_dtm_q['q_index'] = questions.index

    df_dtm_q = df_dtm_q.rename(columns=lambda no: 'q{}'.format(no) if isinstance(no, int) else no)
    df = df_dtm_q.merge(df_dtm_a, right_on='q_index', left_on='q_index')
    indexes_tfidf = df[['q_index', 'a_index']]
    df = df.drop(['q_index', 'a_index'], axis=1)

    dtm = sparse.csr_matrix(df.values)
    sparse.save_npz('./study/data_test/dtm_tfidf_test.npz', dtm)

    # same indexes, so
    indexes_tfidf.to_csv('./study/data_test/indexes.csv', index=False)
