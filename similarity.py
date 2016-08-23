import gensim
import pandas as pd
from annoy import AnnoyIndex
from preprocess_text import text_from_file
from extract import extract_from_url
from collections import defaultdict
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from fast_text import query_fast_text

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

def get_item(item_id):
    # Convert from string to ObjectId:
    return collection.find_one({'_id': ObjectId(item_id)})

def update_integer_id():
    idx = 1
    for document in collection.find():
        document_id = document["_id"]
        modified = document
        modified["integer_id"] = idx
        del(modified["_id"])
        collection.replace_one({"_id":document_id},modified)
        # print(idx)
        # print(document)
        idx += 1

def index_news_doc2vec(index_target_path,
                       doc2vec_model_path="model/doc2vec_400.model",
                       index_dimension=400,tree_size=20):

    doc2vec = gensim.models.Doc2Vec.load(doc2vec_model_path)
    # doc2vec.init_sims(replace=False)
    update_integer_id()
    f = index_dimension
    t = AnnoyIndex(f)
    print("Indexing Doc2Vec")
    counter = 0
    for document in collection.find():
        tokens = document["content"].split()
        vec = doc2vec.infer_vector(tokens)
        # print(document)
        t.add_item(document["integer_id"],vec)
        if counter % 100 == 0:
            print("Doc count:",counter)
        counter += 1

    t.build(tree_size)
    t.save(index_target_path)
    return t

def index_fast_text(index_target_path,tree_size=20):
    update_integer_id()
    f = 100
    t = AnnoyIndex(f)
    print("indexing fast text")
    for document in collection.find():
        vector = query_fast_text(document["content"])
        t.add_item(document["integer_id"],vector[:100])
        print("doc count:", document["integer_id"])
    t.build(tree_size)
    t.save(index_target_path)
    return t

def lda_vector(text,lda_model,dictionary,dimension):
    tokens = text.split()
    doc_bow = dictionary.doc2bow(tokens)
    _formed = np.zeros(dimension)
    _lda = lda_model[doc_bow]
    for lda_idx,val in _lda:
        _formed[lda_idx] = val
    return _formed

def index_news_lda(index_target_path,
                   dict_path="dictionary/all_of_words.dict",
                   lda_model_path="model/lda_100.model",
                   index_dimension=100,tree_size=20):

    dictionary = gensim.corpora.dictionary.Dictionary.load(dict_path)
    lda = gensim.models.ldamulticore.LdaMulticore.load(lda_model_path)

    t = AnnoyIndex(index_dimension)
    update_integer_id()
    print("Indexing LDA")
    for document in collection.find():
        vector = lda_vector(document["content"],lda,dictionary,index_dimension)
        t.add_item(document["integer_id"],vector)
        if document["integer_id"] % 100 == 0:
            print("Doc id:",document["integer_id"])

    t.build(tree_size)
    t.save(index_target_path)
    return t

def compute_nearest_neighbours_lda(path_to_index,
                                   lda_model_path="model/lda_100.model",
                                   dict_path="dictionary/all_of_words.dict",
                                   number_of_nearest_neighbours=10,
                                   index_dimension=100,
                                   show_sentence=True,
                                   verbose=True):
    u = AnnoyIndex(index_dimension)
    u.load(path_to_index)
    lda = gensim.models.ldamulticore.LdaMulticore.load(lda_model_path)
    dictionary = gensim.corpora.dictionary.Dictionary.load(dict_path)
    for document in collection.find():
        vector = lda_vector(document["content"],lda,dictionary,index_dimension)
        sim_idx = u.get_nns_by_vector(vector, number_of_nearest_neighbours)

        nearest_neighbour_ids = []

        for idx in sim_idx:
            neighbour = collection.find_one({"integer_id":idx})
            if neighbour:
                nearest_neighbour_ids.append(neighbour["_id"])

        document_id = document["_id"]
        modified = document
        modified["related_news"] = nearest_neighbour_ids
        del(modified["_id"])
        collection.replace_one({"_id":document_id},modified)

def compute_nearest_neighbours_fast_text(path_to_index,number_of_nearest_neighbours=10):
    f = 100
    u = AnnoyIndex(f)
    u.load(path_to_index)

    for document in collection.find():
        vector = query_fast_text(document["content"])
        sim_idx = u.get_nns_by_vector(vector,number_of_nearest_neighbours)

        nearest_neighbour_ids = []

        for idx in sim_idx:
            neighbour = collection.find_one({"integer_id":idx})
            nearest_neighbour_ids.append(neighbour["_id"])

        document_id = document["_id"]
        modified = document
        del(modified["_id"])
        modified["related_news"] = nearest_neighbour_ids
        collection.replace_one({"_id":document_id},modified)

def compute_nearest_neighbours_doc2vec(path_to_index,
                                       doc2vec_model_path="model/doc2vec_400.model",
                                       number_of_nearest_neighbours=10,
                                       index_dimension=400,
                                       show_sentence=True,
                                       verbose=True):
    f = index_dimension
    u = AnnoyIndex(f)
    u.load(path_to_index)
    doc2vec = gensim.models.Doc2Vec.load(doc2vec_model_path)
    # doc2vec.init_sims(replace=True)
    for document in collection.find():
        tokens = document["content"].split()
        vector = doc2vec.infer_vector(tokens)
        sim_idx = u.get_nns_by_vector(vector, number_of_nearest_neighbours)

        nearest_neighbour_ids = []

        for idx in sim_idx:
            neighbour = collection.find_one({"integer_id":idx})
            nearest_neighbour_ids.append(neighbour["_id"])

        document_id = document["_id"]
        modified = document
        modified["related_news"] = nearest_neighbour_ids
        collection.replace_one({"_id":document_id},modified)

def query_lda_with_file(file_path,lda_model,
                        dictionary,dimension,
                        annoy_index,n=10,
                        include_distances=False):
    text = text_from_file(file_path)
    return query_lda(text,lda_model,
                     dictionary,dimension,
                     annoy_index,n,
                     include_distances)

def query_lda(text,lda_model,
              dictionary,dimension,
              annoy_index,n=10,
              include_distances=False):
    vector = lda_vector(text,lda_model,dictionary,dimension)
    return annoy_index.get_nns_by_vector(vector,n,include_distances=include_distances)

def query_doc2vec(text,doc2vec_model,
                  dictionary,dimension,
                  annoy_index,n=10,
                  include_distances=False):

    vector = doc2vec_model.infer_vector(text.split())
    return annoy_index.get_nns_by_vector(vector,n,include_distances=include_distances)

def query_doc2vec_with_file(file_path,doc2vec_model,
                            annoy_index,n=10,
                            include_distances=False):
    txt = text_from_file(file_path)
    vector = doc2vec_model.infer_vector(txt.split())
    return annoy_index.get_nns_by_vector(vector,n,include_distances=include_distances)

if __name__ == "__main__":
    # index_news("similarity_index/annoy_index",tree_size=300)
    compute_nearest_neighbours("similarity_index/annoy_index")
