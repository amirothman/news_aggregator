from similarity import compute_nearest_neighbours_fast_text,index_fast_text, sort_by_lda_topics, compute_nearest_neighbours_fast_text_with_lda_divergence
from modelling import compute_complete_lda_topics
from crawler import crawl
from fast_text import train_fast_text
import time
import sys

def recalculate():

    print("modelling")
    train_fast_text()

    print("re-indexing")
    index_fast_text("similarity_index/fast_text")

    print("compute lda topics")
    compute_complete_lda_topics("model/lda.model")
    print("get nearest neighours")
    compute_nearest_neighbours_fast_text_with_lda_divergence("similarity_index/fast_text",number_of_nearest_neighbours=50)

total_new_docs = 0

recalculate()

while True:
    print("crawling")
    new_docs = crawl()
    total_new_docs += new_docs
    print(total_new_docs)
    if total_new_docs > 300:
        recalculate()
        total_new_docs = 0
