"""
    Create Document-Thematics matrix and proceed AFC on it
"""
import pickle
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':

    # -------------------------
    # questions/answers indexes
    # -------------------------
    
    #read csv
    
    subforum_answers = pd.read_csv('./data/data_preprocess/subforum_answers.csv')
    subforum_questions= pd.read_csv('./data/data_preprocess/subforum_questions.csv')
    
    index_answers = list(subforum_answers.iloc[:,0])
    index_questions = list(subforum_questions.iloc[:,0])

    data = [item[0] for item in index_questions]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for questions
    P_q = onehot_encoder.fit_transform(np.array(data).reshape(-1, 1))

    data = [item[0] for item in index_answers]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for answers
    P_a = onehot_encoder.fit_transform(np.array(data).reshape(-1, 1))
    
    # -------------------------
    # Thematics-terms matrix
    # -------------------------

    A_count = sparse.load_npz('./data/data_preprocess/dtm_answers_occ.npz')
    Q_count = sparse.load_npz('./data/data_preprocess/dtm_questions_occ.npz')

    TT_q_occ = P_q.transpose().dot(Q_count)
    TT_a_occ = P_a.transpose().dot(A_count)

    A_tfidf = sparse.load_npz('./data/data_preprocess/dtm_answers_tfidf.npz')
    Q_tfidf = sparse.load_npz('./data/data_preprocess/dtm_questions_tfidf.npz')

    TT_q_tfidf = P_q.transpose().dot(Q_tfidf)
    TT_a_tfidf = P_a.transpose().dot(A_tfidf)

    #####
    #save TT Matrix
    ####
    with open('./data/thematics_terms/TTM_questions_occ.pkl', 'wb') as file:
        pickle.dump(TT_q_occ, file)   
    with open('./data/thematics_terms/TTM_answers_occ.pkl', 'wb') as file:
        pickle.dump(TT_a_occ, file)
    with open('./data/thematics_terms/TTM_questions_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_q_tfidf, file)
    with open('./data/thematics_terms/TTM_answers_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_a_tfidf, file)
 
    
    #-------------------------
    #apply AFC to any thematic_term_matrix obtained before, matrix TT are converted to df to csv, 
       #then use csv file in R code (afc.R)
    #--------------------
    
    #read vocabulary file
    with open('./data/data_preprocess/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    
    #TTM_questions_occ
    with open('./data/thematics_terms/TTM_questions_occ.pkl', 'rb') as f:
        TT_q_occ = pickle.load(f)
    
    TT_occ_q = pd.DataFrame.sparse.from_spmatrix(TT_q_occ, columns = vocab)  #shape = 4*m
    TT_occ_q.to_csv('./AFC/TT_occ_q.csv')
       
    #TTM_answers_occ
    with open('./data/thematics_terms/TTM_answers_occ.pkl', 'rb') as f:
        TT_a_occ = pickle.load(f)
    
    TT_occ_a = pd.DataFrame.sparse.from_spmatrix(TT_a_occ, columns = vocab)  #shape = 4*m
    TT_occ_a.to_csv('./AFC/TT_occ_a.csv')
    
    #TTM_questions_tfidf
    with open('./data/thematics_terms/TTM_questions_tfidf.pkl', 'rb') as f:
        TT_q_tfidf = pickle.load(f)
    
    TT_tfidf_q = pd.DataFrame.sparse.from_spmatrix(TT_q_tfidf, columns = vocab)  #shape = 4*m
    TT_tfidf_q.to_csv('./AFC/TT_tfidf_q.csv')
    
    #TTM_answers_tfidf
    with open('./data/thematics_terms/TTM_answers_tfidf.pkl', 'rb') as f:
        TT_a_tfidf = pickle.load(f)
    
    TT_tfidf_a = pd.DataFrame.sparse.from_spmatrix(TT_a_tfidf, columns = vocab)  #shape = 4*m
    TT_tfidf_a.to_csv('./AFC/TT_tfidf_a.csv')
    
    