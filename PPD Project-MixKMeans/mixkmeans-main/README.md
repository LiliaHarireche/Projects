# mixkmeans

University Project for Master's degree in Machine Learning for Data Science (Université de Paris)

Antoine Rodriguez - [Lilia Harireche](https://github.com/LiliaHarireche)

---------------------------

The main objective is to reproduce some results from \
*MixKMeans: Clustering Question-Answer Archives, Deepak P, 2016* 
([see here](https://www.researchgate.net/publication/312251324_MixKMeans_Clustering_Question-Answer_Archives)) \
and propose an implementation of the algorithm **MixKMeans**.


We have also reproduce some results from \
*CQADupStack: A Benchmark Data Set for Community Question-Answering Research, Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy, 2015* 
([see here](https://www.researchgate.net/publication/296699072_CQADupStack_A_Benchmark_Data_Set_for_Community_Question-Answering_Research)) \
GitHub : [CQADupStack](https://github.com/D1Doris/CQADupStack)


---------------------------
Original data from **CQADupStack** are downloadable [here](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

---------------------------

Codes are explained in their directories \
`training.py` is the main script to run our trainings of differents versions of MixKMeans \

`/tests/` contains unitary tests **[NOT UPDATED]**

---------------------------

Data and all data preprocess results are too heavy for GitHub, we have locally on root :
```
data
└───data_preprocess
│      dtm matrix, index of QA
|      other informations like scores
|      vocabulary
└───original_data
       *.json
```

