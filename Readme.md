# Clustering News With Artificial Intelligence And Lots of Love

This is an experimental project to cluster news articles. Some of the technologies used include:

Text modelling:
  - word2vec (gensim)
  - doc2vec (gensim)
  - fastText
  - LDA (gensim)

Database:
  - redis
  - mongodb

Web back-end:
  - Flask

Nearest-neighbour Approximation:
  - Annoy

## Notes

Currently, the document processing is a bit slow ( ~10 Minutes for ~3000 articles).

## ''Installation''

    git clone https://github.com/amirothman/news_aggregator

install missing dependencies with

    pip install <package-name>

create following empty-directories if they do not exist:

    model
    corpus
    similarity_index
    textfiles

## Launching the web server

With flask:

    FLASK_APP=webapp.py flask run

With Gunicorn:

    gunicorn -b 0.0.0.0:5000 webapp:app
