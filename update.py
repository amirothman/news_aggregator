from corpus_dictionary import recreate_dictionary
from modelling import update_doc2vec_model,google_news_model,update_lda_model
from similarity import index_news_doc2vec,compute_nearest_neighbours_doc2vec,index_news_lda,compute_nearest_neighbours_lda,compute_nearest_neighbours_fast_text,index_fast_text, compute_sub_lda_topics
from crawler import crawl
from fast_text import train_fast_text
import time

def recalculate():

    print("modelling")
    train_fast_text()

    print("re-indexing")
    index_fast_text("similarity_index/fast_text")

    print("get nearest neighours")
    compute_nearest_neighbours_fast_text("similarity_index/fast_text",number_of_nearest_neighbours=300)

    # print("modelling")
    # update_doc2vec_model()
    #
    # print("re-indexing")
    # index_news_doc2vec("similarity_index/doc2vec")

    # print("compute nearest neighbours doc2vec")
    # compute_nearest_neighbours_doc2vec("similarity_index/doc2vec",number_of_nearest_neighbours=200)

    print("compute sub lda topics")
    compute_sub_lda_topics(related_key="related_news_fast_text")
    #

#


total_new_docs = 0
#
recalculate()
while True:
    print("crawling")
    new_docs = crawl()
    total_new_docs += new_docs
    print(total_new_docs)
    if total_new_docs > 300:
        recalculate()
        total_new_docs = 0
