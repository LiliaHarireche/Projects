"""

"""
import math
from random import randint
from random import sample
from time import time

import pickle

from scipy import sparse
import numpy as np

from mixkmeans.distances import composite_distance


class MixKMeans:
    def __init__(self, x, weights, distance='eucl'):
        """
        Initialize the MixKMeans model with hyper-parameters used in distance
        computations

        :param x: negative float
        :param weights: tuple or list of the two weights given to each part
        """
        if x != 1 and x > 0:
            raise ValueError('x must be negative or zero')
        else:
            self.x = x  # huge negative

        if weights[0] + weights[1] != 1:
            raise ValueError('Sum of weights must be one')
        else:
            self.weights = weights

        self.K = None
        self.distance = distance

        self.cost_historic = None
        self.prototypes = None  # best prototypes
        self.iteration = 0

    # --------------
    # - Methods needed to fit the model to the data
    # --------------
    def alternative_init(self, dataset, K):
        """
        Initialize prototypes (i.e. centroid in this mixkmeans) inspired to 'spreading out the cluster centroids'
        :param dataset:
        :param K: number of clusters
        """
        indexes = [randint(0, dataset.shape[0] - 1)]

        for i in range(K - 1):
            vec_dist = []
            for row in dataset:
                sum_dist = 0
                for ind in indexes:
                    sum_dist += composite_distance(row, dataset[ind], self.x, self.weights, self.distance)
                vec_dist.append(sum_dist)

            indexes.append(np.argmax(vec_dist))

        prototypes = []
        for ind in indexes:
            prototypes.append(dataset[ind])
        self.prototypes = prototypes

    def initialize_prototypes(self, dataset, K):
        """Random initialization of prototypes"""
        indexes = sample(range(dataset.shape[0]), K)
        prototypes = []
        for ind in indexes:
            prototypes.append(dataset[ind])
        self.prototypes = prototypes

    def assign_clusters(self, dataset):
        """
        Assign each point of the dataset to a cluster defines by its prototype

        :param dataset:
        :param prototypes: list of indexes of prototypes
        :return:
        """
        assignation = []
        for row in dataset:
            distances = []
            for index, prototype in enumerate(self.prototypes):
                distances.append(composite_distance(row, prototype, self.x, self.weights, self.distance))
            assignation.append(np.argmin(distances))

        return assignation

    def compute_prototypes(self, dataset, assignation):
        """
        Compute prototypes for each cluster
        :return:
        """
        prototypes = []
        for cluster_ind in range(self.K):

            indexes = np.where(np.array(assignation) == cluster_ind)[0]  # indexes where Q | A in current cluster
            Q = dataset[indexes, 0:int(dataset.shape[1] / 2)]
            A = dataset[indexes, int(dataset.shape[1] / 2):]

            sum_dist_q = 0.000001
            sum_dist_a = 0.000001

            if dataset[indexes].shape[0] == 0:
                prototypes.append(sparse.csr_matrix(np.zeros((1, dataset.shape[1]))))
                continue

            for index, row in enumerate(dataset[indexes]):
                c_d = composite_distance(row, self.prototypes[cluster_ind], self.x, self.weights, self.distance)
                if c_d != 0:
                    try:
                        distance_qa = math.pow(c_d, (1 - self.x)/self.x)
                    except ValueError:
                        distance_qa = 0
                else:
                    distance_qa = 0
                dist_q = distance_qa * composite_distance(row, self.prototypes[cluster_ind], self.x - 1, (1, 0), self.distance)
                dist_a = distance_qa * composite_distance(row, self.prototypes[cluster_ind], self.x - 1, (0, 1), self.distance)

                Q[index] = Q[index].multiply(dist_q)
                A[index] = A[index].multiply(dist_a)

                sum_dist_q += dist_q
                sum_dist_a += dist_a

            Q = Q.multiply(1 / sum_dist_q)
            A = A.multiply(1 / sum_dist_a)

            qa = np.concatenate([np.array(Q.sum(axis=0)), np.array(A.sum(axis=0))], axis=1)
            prototypes.append(sparse.csr_matrix(qa))

        self.prototypes = prototypes

    # --------------
    # - fit and predict
    # --------------
    def fit(self, dataset, K, itermax):
        """
        Process training of MixKmeans model on our dataset

        :param dataset: a sparse matrix with Q | A in rows
        :param K: number of clusters
        :return:
        """
        print('Begin fitting')
        self.K = K

        # if needed
        if not self.cost_historic:
            self.cost_historic = []
        if not self.prototypes:
            t = time()
            self.initialize_prototypes(dataset, self.K)
            print('initalisation :  {} s'.format(time() - t))

        assignation = self.assign_clusters(dataset)

        old_cost = 0
        condition = True
        while (self.iteration < itermax) & condition:
            # calcul de la fonction coût equivalent à l'inertie intraclasse
            t = time()
            cost = 0
            for ind in range(self.K):
                mat = dataset[np.array(assignation) == ind]
                for row in mat:
                    c_d = composite_distance(row, self.prototypes[ind], self.x, self.weights, self.distance)
                    if c_d != 0:
                        try:
                            cost += math.pow(c_d, 1 / self.x)
                        except ValueError:
                            cost += 0

            condition = (abs(cost-old_cost) >= 0.001)  # boolean

            self.compute_prototypes(dataset, assignation)
            assignation = self.assign_clusters(dataset)

            self.iteration += 1
            print('ITERATION {}  :  {} s'.format(self.iteration, time() - t))
            self.cost_historic.append(cost)
            old_cost = cost

        if self.iteration >= itermax:
            print('Pas de convergence ! Processus arrêté au bout de {} iterations)'.format(self.iteration))  # english

        return cost

    def predict(self, dataset):
        """can predict on qa in csr_matrix form"""
        if self.prototypes:
            assignation = self.assign_clusters(dataset)
            return assignation
        else:
            raise TypeError('Need to fit the model before')

    # --------------
    # - save
    # --------------
    def save_state(self, save_file='test.pkl'):  # test.pkl for unitary test
        with open(save_file, 'wb') as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    print('coucou')
