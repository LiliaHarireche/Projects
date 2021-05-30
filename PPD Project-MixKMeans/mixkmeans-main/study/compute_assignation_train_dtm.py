import pickle

from scipy import sparse
import numpy as np

from study.utils import get_question_cluster, update_model


if __name__ == '__main__':
    # occ eucl
    with open('results_training/models/occ_eucl100_best.pkl', 'rb') as file:
        model = pickle.load(file)

    model = update_model(model)

    DTM = sparse.load_npz('data/data_preprocess/dtm_occ.npz')
    assignation = get_question_cluster(DTM, model)
    np.savez('study/assignations/occ_eucl_asgn.npz', assignation)

    # tfidf eucl
    with open('results_training/models/tfidf_eucl100_best.pkl', 'rb') as file:
        model = pickle.load(file)

    model = update_model(model)

    DTM = sparse.load_npz('data/data_preprocess/dtm_tfidf.npz')
    assignation = get_question_cluster(DTM, model)
    np.savez('study/assignations/tfidf_eucl_asgn.npz', assignation)

    # tfidf cosin
    with open('results_training/models/tfidf_cosin100_best.pkl', 'rb') as file:
        model = pickle.load(file)

    model = update_model(model)

    DTM = sparse.load_npz('data/data_preprocess/dtm_tfidf.npz')
    assignation = get_question_cluster(DTM, model)
    np.savez('study/assignations/tfidf_cosin_asgn.npz', assignation)
