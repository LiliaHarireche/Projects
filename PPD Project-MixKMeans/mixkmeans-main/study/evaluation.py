import pickle
from time import time
import argparse
import os

from scipy import sparse
import numpy as np

from study.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-v", "-value", help=".")

args = parser.parse_args()

if __name__ == '__main__':
    # ## OCC EUCL
    with open('results_training/models/tfidf_cosin100_best.pkl', 'rb') as file:
        model = pickle.load(file)
    assignation = np.load('study/assignations/occ_eucl_asgn.npz')['arr_0']
    dtm = sparse.load_npz('data/data_preprocess/dtm_occ.npz')
    model_new = update_model(model)
    DTM = sparse.load_npz('study/data_test/dtm_occ_test.npz')

    precisions = []
    recalls = []
    F_scores = []
    for qa in DTM[(args.value - 100):args.value]:
        t = time()
        ind_cluster = get_question_cluster(qa, model_new)
        index_closest_q = get_closest_question(qa, ind_cluster, dtm, assignation, 'eucl')

        scores = get_scores(qa, dtm[index_closest_q])

        precisions.append(scores[0])
        recalls.append(scores[1])
        F_scores.append(scores[2])
        print('tmp exec : %s' % (time() - t))

    np.savez('study/scores_occ_eucl100_{}.npz'.format(args.value), P=precisions, R=recalls, F=F_scores)

    # ## OCC COSIN

    # ## TFIDF COSIN
