import gensim
import pandas as pd
from annoy import AnnoyIndex
from preprocess_text import text_from_file
from extract import extract_from_url
from collections import defaultdict
from pymongo import MongoClient
from bson.objectid import ObjectId

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
        collection.replace_one({"_id":document_id},modified)
        # print(idx)
        # print(document)
        idx += 1

def index_news(index_target_path,
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
        print("Doc count:",counter)
        counter += 1

    t.build(tree_size)
    t.save(index_target_path)
    return t

def compute_nearest_neighbours(path_to_index,
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
