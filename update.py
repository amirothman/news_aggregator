from modelling import update_doc2vec_model,google_news_model
from similarity import index_news,compute_nearest_neighbours
from crawler import crawl
import time


size=1000

update_doc2vec_model(size=size,min_count=5)
# google_news_model()
print("re-indexing")
index_news("similarity_index/annoy_index",tree_size=100,index_dimension=size)
print("get nearest neighours")
compute_nearest_neighbours("similarity_index/annoy_index",index_dimension=size)


#
# total_new_docs = 0
#
# while True:
#     print("crawwling")
#     new_docs = crawl()
#     total_new_docs += new_docs
#     print(total_new_docs)
#     if total_new_docs > 300:
#         print("update model")
#         update_doc2vec_model()
#         print("re-indexing")
#         index_news("similarity_index/annoy_index",tree_size=300)
#         print("get nearest neighours")
#         compute_nearest_neighbours("similarity_index/annoy_index")
#         total_new_docs = 0
#
#     print("sleeping")
#     time.sleep(500)
