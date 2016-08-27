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
    # compute_nearest_neighbours_fast_text("similarity_index/fast_text",number_of_nearest_neighbours=10)

    compute_nearest_neighbours_fast_text_with_lda_divergence("similarity_index/fast_text",number_of_nearest_neighbours=20)

print("compute lda topics")
compute_complete_lda_topics("model/lda.model")
while True:
    recalculate()
