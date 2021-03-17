Dans ce projet, les jeux de données utilisés 
n'ont pas pu être chargés vu leur grand volume.

Le jeu de données initial est un fichier .txt contenant 
des informations sur différents articles, ci-dessous 
un exemple d'un article:
--------
#*Formal models for expert finding in enterprise corpora.
#@Krisztian Balog,Leif Azzopardi,Maarten de Rijke
#t2006
#cSIGIR
#index594377
#%595386
#%362694
#%772628
#%595551
#%26506
#%594777
#%935966
#%121844
#%596024
#%95047
#%595671
#!Searching an organization's document repositories for experts provides a cost effective solution for the task of expert finding. We present two general strategies to expert searching given a document collection which are formalized using generative probabilistic models. The first of these directly models an expert's knowledge based on the documents that they are associated with, whilst the second locates documents on topic, and then finds the associated expert. Forming reliable associations is crucial to the performance of expert finding systems. Consequently, in our evaluation we compare the different approaches, exploring a variety of associations along with other operational parameters (such as topicality). Using the TREC Enterprise corpora, we show that the second strategy consistently outperforms the first. A comparison against other unsupervised techniques, reveals that our second model delivers excellent performance.
-------

Par la suite 8 csv ont été créés afin d'établir un tableau de bord
en se basant sur le prétraitement du texte ci-dessus pour extraire
toutes les informations, par exemple: titre, auteurs, résumé,
références,...

