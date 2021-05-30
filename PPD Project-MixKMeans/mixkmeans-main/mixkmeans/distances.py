""""""
import math


def dist_eucl(a, b):
    """Element-wise distance between two sparce vector 1xM"""
    if a.shape == b.shape:
        return (a - b).power(2).sum(axis=1)[0, 0]
    else:
        raise ValueError('a and b must have the same shape')


def dist_cosin(a, b):
    """cosinus similarity between two sparce vector 1xM"""
    if a.shape == b.shape:
        return 1 - (a * b.transpose())[0, 0] / ((a * a.transpose())[0, 0] * (b * b.transpose())[0, 0] + 0.0001)
    else:
        raise ValueError('a and b must have the same shape')


# POINT = (question vectorisée, reponse vectorisée)
def composite_distance(point, prototype, x, weights, distance='eucl'):
    """Compute  point-to-prototype (or point-to-point) distance"""
    if distance == 'eucl':
        distance_func = dist_eucl
    elif distance == 'cosin':
        distance_func = dist_cosin
    else:
        raise ValueError('unknown distance')

    if point.shape == prototype.shape:
        if point.shape[1] % 2 == 0:
            d1 = distance_func(point[:, 0:int(point.shape[1] / 2)], prototype[:, 0:int(prototype.shape[1] / 2)])
            d2 = distance_func(point[:, int(point.shape[1] / 2):], prototype[:, int(prototype.shape[1] / 2):])

            temp = 0
            if weights[0] * d1 != 0:
                temp += math.pow(weights[0] * d1, x)
            if weights[1] * d2 != 0:
                temp += math.pow(weights[1] * d2, x)
            return temp
        else:
            raise ValueError('Length of vectors must be even')  # by construction
    else:
        raise ValueError('point and prototype must have the same shape')


if __name__ == '__main__':
    pass
