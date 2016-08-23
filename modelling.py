from gensim import corpora,models,utils
import numpy as np
from pymongo import MongoClient
from preprocess_text import clean

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

def tagged_docs():
    for d in collection.find():
        # cleaned = clean(d["content"])
        doc_id = str(d["_id"])
        yield models.doc2vec.TaggedDocument(d["content"].split(),[doc_id])

def update_doc2vec_model(model_path="model/doc2vec_400.model",size=400,min_count=5):
    doc2vec = models.doc2vec.Doc2Vec(tagged_docs(), size=size, window=8, min_count=min_count, workers=6)
    doc2vec.save(model_path)
    return doc2vec

def update_lda_model(model_path="model/lda_100.model",size=100,corpus_path='corpus/all_of_words.mm'):
    corpus = corpora.mmcorpus.MmCorpus(corpus_path)
    print("LDA Topic Modelling")
    lda = models.ldamulticore.LdaMulticore(corpus,num_topics=size,eta="auto",workers=6)
    lda.save(model_path)
    return lda

def google_news_model(model_target_path="model/google_news_model.model",model_source_path="/home/amir/makmal/ular_makan_surat_khabar/word_embeddings/GoogleNews-vectors-negative300.bin"):
    doc2vec = models.doc2vec.Doc2Vec(min_count=10)
    doc2vec.load_word2vec_format(model_source_path,binary=True)
    # doc2vec.init_sims(replace=False)
    doc2vec.save(model_target_path)
    return doc2vec

if __name__ == "__main__":
    # for el in quranic_sentences_with_docs():
    #     print(el)
    print("doc2vec 400")
    update_doc2vec_model()
    # quran_dictionary = corpora.dictionary.Dictionary.load("dictionary/quranic_ayat_en.dict")


    # LDA Topic Modelling


    # Doc2Vec Modelling
    #
    # print("doc2vec modelling")
