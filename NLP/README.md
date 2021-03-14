# gender-bias-repo

Reproduction of *Evaluating Gender Bias in Machine Translation Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer, (ACL 2019)* 
(original repository [mt_gender](https://github.com/gabrielStanovsky/mt_gender))

University Project for Master's degree in **M**achine **L**earning for **D**ata **S**cience (Université de Paris)
Antoine Rodriguez - Lilia Harireche - Douaa Benhaddouche

-----------

**Install requirements**
```
bash install.sh
```

**Reproduce results**
```
python ./scripts/main.py
```

Translations were made by original authors and put in ./languages.
An attempt is in `translate.py` and `translate_ API.py`  but it need to configure APIs.

-----------
**Create a Docker container of the repo** 

In gender-bias-repo root :
```
docker build -t gender-bias-docker .
```
then
```
docker run -t gender-bias-docker
```
