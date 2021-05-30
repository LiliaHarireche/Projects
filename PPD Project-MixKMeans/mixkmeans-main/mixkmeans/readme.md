## mixkmeans

Main module that contains : 

- `mixkmeans.py` where the `MixKMeans()` class is built : \
  MixKMeans takes 3 arguments x, weights and distance\
  `fit()` method trains the model with a dataset (DTM) and the number of clusters K \
  `predict()` method classifies a new piece of data in a cluster
  
- `distances.py` contains distance used by MixKMeans named `composite_distance` which changes with original distance ( euclidean or cosinus) and hyper-parameters x and weights.