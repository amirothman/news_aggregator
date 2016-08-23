from corpus_dictionary import recreate_dictionary
from modelling import update_doc2vec_model,google_news_model,update_lda_model
from similarity import index_news,compute_nearest_neighbours,index_news_lda,compute_nearest_neighbours_lda,compute_nearest_neighbours_fast_text,index_fast_text
from crawler import crawl
from fast_text import train_fast_text
import time

print("modelling")
# train_fast_text()

print("re-indexing")
index_fast_text("similarity_index/fast_text")

print("get nearest neighours")
compute_nearest_neighbours_fast_text("similarity_index/fast_text")

#
# total_new_docs = 0
# #
# while True:
#     print("crawwling")
#     new_docs = crawl()
#     total_new_docs += new_docs
#     print(total_new_docs)
#     if total_new_docs > 300:
#         print("recreate corpus")
#         recreate_dictionary()
#
#         print("modelling")
#         update_lda_model()
#
#         print("re-indexing")
#         index_news_lda("similarity_index/annoy_index_lda")
#
#         print("get nearest neighours")
#         compute_nearest_neighbours_lda("similarity_index/annoy_index_lda")
#
#     print("sleeping")
#     time.sleep(500)
