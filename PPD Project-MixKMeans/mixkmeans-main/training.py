"""
    launch locally (jupyter notebook)
"""
from scipy import sparse

import numpy as np

from mixkmeans import MixKMeans


def launch(data, distance, save_model, save_cost):
    dtm = sparse.load_npz(data)

    print('begin')
    model = MixKMeans(x=-3, weights=(0.2, 0.8), distance=distance)
    cost = model.fit(dtm, 100, 50)

    model.save_state(save_file=save_model)
    np.savez(save_cost, cost)


if __name__ == '__main__':
    pass
