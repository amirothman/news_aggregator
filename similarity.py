import gensim
import pandas as pd
from annoy import AnnoyIndex
from preprocess_text import text_from_file,clean
from extract import extract_from_url
from collections import defaultdict
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from fast_text import query_fast_text
from modelling import mini_lda_model
from scipy.stats import entropy
from numpy.linalg import norm
import subprocess
import redis

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']


def jensen_shannon_divergence(P, Q):
    """http://stackoverflow.com/questions/15880133/jensen-shannon-divergence"""
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

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
        idx += 1

def index_news_doc2vec(index_target_path,
                       doc2vec_model_path="model/doc2vec.model",
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

def fast_text_bulk():
    subprocess.run(["sh","fast_text_bulk.sh"])

def fast_text_vector(text,dictionary):
    container = []
    for word in text.split():
        try:
            container.append(dictionary[word])
        except KeyError:
            pass
    return np.sum(container,axis=0)/len(container)

def fast_text_vector_from_redis(text,redis_connection):
    container = []
    for word in text.split():
        str_word_vector = redis_connection.get(word)
        if str_word_vector:
            lst_word_vector = [float(v) for v in str_word_vector.split()]
            if len(lst_word_vector) == 100:
                np_array_word_vector = np.array(lst_word_vector)
                container.append(np_array_word_vector)

    if len(container) == 0:
        return np.zeros(100)
    else:
        return np.sum(container,axis=0)/len(container)

def fast_text_dictionary():
    temp_dict = {}
    with open("textfiles/fasttext_word_vector.txt","r") as word_vector_file:
        for line in word_vector_file:
            splitted = line.split()
            list_string = splitted[1:]
            if len(list_string) == 100:
                as_vector = [float(v) for v in list_string]
                temp_dict[splitted[0]] = np.array(as_vector)
    return temp_dict

def fast_text_to_redis(redis_connection):

    with open("textfiles/fasttext_word_vector.txt","r") as word_vector_file:
        for line in word_vector_file:
            splitted = line.split()
            word = splitted[0]
            list_string = splitted[1:]
            if len(list_string) == 100:
                redis_connection.set(word," ".join(splitted[1:]))

def index_fast_text(index_target_path,tree_size=20):
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    update_integer_id()
    f = 100
    t = AnnoyIndex(f)
    print("writing raw text to temp file")
    with open("textfiles/temp_fast_text","w") as temp_file:
        counter = 0
        for document in collection.find():
            counter += 1
            temp_file.write(document["content"])
            temp_file.write("\n")

    print("build fastText vectors")
    fast_text_bulk()
    print("transfer to redis")
    fast_text_to_redis(r)

    print("indexing fast text")
    for document in collection.find():
        # print(document)
        vector = fast_text_vector_from_redis(document["content"],r)
        t.add_item(document["integer_id"],vector[:100])
        if document["integer_id"] % 100 == 0:
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
                   lda_model_path="model/lda.model",
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
                                   lda_model_path="model/lda.model",
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

def compute_nearest_neighbours_fast_text(path_to_index,number_of_nearest_neighbours=200):
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    f = 100
    u = AnnoyIndex(f)
    u.load(path_to_index)

    for document in collection.find():
        vector = fast_text_vector_from_redis(document["content"],r)
        sim_idx = u.get_nns_by_vector(vector,number_of_nearest_neighbours)
        nearest_neighbour_ids = []
        if document["integer_id"] % 100 == 0:
            print(document["integer_id"])
        for idx in sim_idx:
            neighbour = collection.find_one({"integer_id":idx})
            if neighbour:
                nearest_neighbour_ids.append(neighbour["_id"])

        document_id = document["_id"]
        modified = document
        del(modified["_id"])
        modified["related_news_fast_text"] = nearest_neighbour_ids
        collection.replace_one({"_id":document_id},modified)

def compute_nearest_neighbours_doc2vec(path_to_index,
                                       doc2vec_model_path="model/doc2vec.model",
                                       number_of_nearest_neighbours=200,
                                       index_dimension=400,
                                       show_sentence=True,
                                       verbose=True):
    f = index_dimension
    u = AnnoyIndex(f)
    u.load(path_to_index)
    doc2vec = gensim.models.Doc2Vec.load(doc2vec_model_path)
    for document in collection.find():
        cleaned = clean(document["content"])
        tokens = cleaned.split()
        vector = doc2vec.infer_vector(tokens)
        sim_idx = u.get_nns_by_vector(vector, number_of_nearest_neighbours)
        if verbose:
            print("doc_id: ",document["integer_id"])
        nearest_neighbour_ids = []

        for idx in sim_idx:
            neighbour = collection.find_one({"integer_id":idx})
            nearest_neighbour_ids.append(neighbour["_id"])

        document_id = document["_id"]
        modified = document
        del(modified["_id"])
        modified["related_news_doc2vec"] = nearest_neighbour_ids
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

def compute_sub_lda_topics(related_key="related_news_doc2vec"):
    i = 0
    for document in collection.find():
        document_id = document["_id"]
        print(i)
        i += 1
        subcollection = []
        for d in document[related_key]:
            doc = collection.find_one({"_id":d})
            subcollection.append(doc)

        dimension = 10
        # print(subcollection)
        lda_model,dictionary = mini_lda_model(subcollection,num_topics=dimension)
        doc_vector = lda_vector(clean(document["content"]),lda_model,dictionary,dimension)
        lda_jensen_shannon_divergences = []
        for related in subcollection:
            cleaned = clean(related["content"])
            related_vector = lda_vector(cleaned,lda_model,dictionary,dimension)
            lda_jensen_shannon_divergences.append((jensen_shannon_divergence(doc_vector,related_vector),related["_id"]))

        sorted_related = sorted(lda_jensen_shannon_divergences,key = lambda x: x[0])
        sorted_divergence = [s[0] for s in sorted_related ]
        sorted_object_id_by_divergence = [s[1] for s in sorted_related ]
        # print(sorted_related)
        modified = document
        del(modified["_id"])
        modified["lda_jensen_shannon_divergences"] = sorted_divergence
        modified["object_id_by_divergence"] = sorted_object_id_by_divergence
        collection.replace_one({"_id":document_id},modified)

def sort_by_lda_topics(lda_model,dictionary,related_key,dimension=100):
    i = 0
    for document in collection.find():
        document_id = document["_id"]
        if i%100 == 0:
            print(i)
        doc_vector = lda_vector(clean(document["content"]),lda_model,dictionary,dimension)

        related_documents = []
        for d in document[related_key]:
            doc = collection.find_one({"_id":d})
            related_documents.append(doc)

        lda_jensen_shannon_divergences = []
        for doc in related_documents:
            cleaned = clean(related["content"])
            related_vector = lda_vector(cleaned,lda_model,dictionary,dimension)
            lda_jensen_shannon_divergences.append((jensen_shannon_divergence(doc_vector,related_vector),related["_id"]))

        sorted_related = sorted(lda_jensen_shannon_divergences,key = lambda x: x[0])
        sorted_divergence = [s[0] for s in sorted_related ]
        sorted_object_id_by_divergence = [s[1] for s in sorted_related ]
        # print(sorted_related)
        modified = document
        del(modified["_id"])
        modified["lda_jensen_shannon_divergences"] = sorted_divergence
        modified["object_id_by_divergence"] = sorted_object_id_by_divergence
        collection.replace_one({"_id":document_id},modified)

if __name__ == "__main__":
    # r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # fast_text_to_redis(r)
    compute_sub_lda_topics(related_key="related_news_fast_text")
