import unittest

from scipy import sparse

from mixkmeans.distances import (
    dist_eucl,
    dist_cosin,
    composite_distance,
)

from mixkmeans import MixKMeans


class DistancesTest(unittest.TestCase):
    def setUp(self):
        self.dtm = sparse.load_npz('dtm_test.npz')

    @unittest.skip
    def test_dist(self):
        print(dist_eucl(self.dtm[0], self.dtm[42]))

    @unittest.skip
    def test_dist_cosin(self):
        print(dist_cosin(self.dtm[0], self.dtm[42]))

    def test_composite_distance(self):
        print(composite_distance(self.dtm[0], self.dtm[86], -3, (0.2, 0.8)))


if __name__ == '__main__':
    unittest.main()
