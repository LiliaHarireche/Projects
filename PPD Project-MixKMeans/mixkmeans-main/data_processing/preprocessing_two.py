"""
    Construction of concatenated matrix Q | A and index
"""
import os

import pandas as pd

from scipy import sparse

df = pd.read_csv('./data/data_preprocess/indexes.csv')

if __name__ == '__main__':
    os.chdir('../../')
    # Occurences matrix
    answers = pd.read_csv('data/data_preprocess/subforum_answers.csv', index_col=0)
    questions = pd.read_csv('data/data_preprocess/subforum_questions.csv', index_col=0)

    dtm_a = sparse.load_npz('data/data_preprocess/dtm_answers_occ.npz')
    dtm_q = sparse.load_npz('data/data_preprocess/dtm_questions_occ.npz')

    df_dtm_a = pd.DataFrame.sparse.from_spmatrix(dtm_a)
    df_dtm_q = pd.DataFrame.sparse.from_spmatrix(dtm_q)

    df_dtm_a['a_index'] = answers.index   # useless
    df_dtm_a['q_index'] = list(answers['parentid'])
    df_dtm_q['q_index'] = questions.index

    df_dtm_q = df_dtm_q.rename(columns=lambda no: 'q{}'.format(no) if isinstance(no, int) else no)
    df = df_dtm_q.merge(df_dtm_a, right_on='q_index', left_on='q_index')
    indexes_occ = df[['q_index', 'a_index']]
    df = df.drop(['q_index', 'a_index'], axis=1)

    dtm = sparse.csr_matrix(df.values)
    sparse.save_npz('./data/data_preprocess/dtm_occ.npz', dtm)

    # tf_idf matrix
    answers = pd.read_csv('data/data_preprocess/subforum_answers.csv', index_col=0)
    questions = pd.read_csv('data/data_preprocess/subforum_questions.csv', index_col=0)

    dtm_a = sparse.load_npz('data/data_preprocess/dtm_answers_tfidf.npz')
    dtm_q = sparse.load_npz('data/data_preprocess/dtm_questions_tfidf.npz')

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
    sparse.save_npz('./data/data_preprocess/dtm_tfidf.npz', dtm)

    # same indexes, so
    indexes_tfidf.to_csv('./data/data_preprocess/indexes.csv', index=False)
