import pickle

from scipy import sparse
import numpy as np

from mixkmeans.mixkmeans import MixKMeans
from mixkmeans.distances import composite_distance


def update_model(mkm_object):
    """
    Set an object MixKMeans for use of predict() method
    model was save as a pkl file and we use prototypes
    this function is needed to avoid compatibility problems !
    :param mkm_object:
    """
    model = MixKMeans(-3, (0.2, 0.8), distance=mkm_object.distance)  # the latest version of the algorithm
    model.prototypes = mkm_object.prototypes
    return model


def get_question_cluster(qa, model):
    """/!\ return a list"""
    return model.predict(qa)[0]


def get_closest_question(qa, cluster_id, dtm_train, assignation, distance):
    # on récupère les q,a de ces clusters
    dtm = dtm_train[assignation == cluster_id]
    index_closest = 0
    smallest_distance = 100000000000
    for ind, item in enumerate(dtm):
        if composite_distance(qa, item, -3, (0.2, 0), distance) <= smallest_distance:
            index_closest = ind

    return index_closest


def get_scores(qa, qa_pred):
    a = [1 if item != 0 else item for item in qa[:, int(qa.shape[1] / 2):].A.tolist()[0]]
    a_pred = [1 if item != 0 else item for item in qa_pred[:, int(qa_pred.shape[1] / 2):].A.tolist()[0]]

    nb_terms_gold = sum(a)
    nb_terms_pred = sum(a_pred)
    nb_communs = [x + y for x, y in zip(a, a_pred)].count(2)

    if nb_terms_pred != 0:
        precision = nb_communs / nb_terms_pred
    else:
        precision = 0
    if nb_terms_gold !=0:
        recall = nb_communs / nb_terms_gold
    else:
        recall = 0
    if precision + recall != 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0
    return precision, recall, f_score


if __name__ == '__main__':
    # TEST
    with open('results_training/models/occ_eucl100_best.pkl', 'rb') as file:
        model = pickle.load(file)

    assignation = np.load('study/assignations/occ_eucl_asgn.npz')['arr_0']
    dtm = sparse.load_npz('data/data_preprocess/dtm_occ.npz')

    model_new = update_model(model)

    DTM = sparse.load_npz('study/data_test/dtm_occ_test.npz')
    qa = DTM[0]   # boucle ici
    ind_cluster = get_question_cluster(qa, model_new)
    index_closest_q = get_closest_question(qa, ind_cluster, dtm, assignation, 'eucl')

    scores = get_scores(qa, dtm[index_closest_q])


