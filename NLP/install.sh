#!/bin/bash
pip install -r ./requirements.txt
python -m spacy download "de_core_news_sm"
python -m spacy download "fr_core_news_sm"
python -m spacy download "it_core_news_sm"
python -m spacy download "es_core_news_sm"
echo "DONE"
