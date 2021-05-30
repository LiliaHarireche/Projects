## preprocessing

- `utils.py` contains the class `SubForum` made to facilitate preprocessing.

- `preprocessing_one.py` describes the first part of preprocessing and the construction of the DTMs (document-terms matrix) \
- `preprocessing_two.py` constructs final DTMs `dtm_occ.npz` and `dtm_tfidf.npz` in `scipy.sparse.csr_matrix` format.