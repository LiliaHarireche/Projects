FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r ./requirements.txt
RUN python -m spacy download "de_core_news_sm"
RUN python -m spacy download "fr_core_news_sm"
RUN python -m spacy download "it_core_news_sm"
RUN python -m spacy download "es_core_news_sm"

RUN mkdir src
RUN mkdir src/results
WORKDIR /src
COPY . .

CMD [ "python", "./scripts/main.py" ]


