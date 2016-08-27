from gensim import corpora,models,utils
import numpy as np
from pymongo import MongoClient
from preprocess_text import clean
from corpus_dictionary import custom_corpus, CompleteCorpus

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

def tagged_docs():
    for d in collection.find():
        cleaned = clean(d["content"])
        doc_id = str(d["_id"])
        yield models.doc2vec.TaggedDocument(cleaned.split(),[doc_id])

def update_doc2vec_model(model_path="model/doc2vec.model",size=400,min_count=5):
    doc2vec = models.doc2vec.Doc2Vec(tagged_docs(), size=size, window=8, min_count=min_count, workers=6)
    doc2vec.save(model_path)
    return doc2vec

def update_lda_model(corpus,model_path="model/lda.model",size=100):
    print("LDA Topic Modelling")
    lda = models.ldamulticore.LdaMulticore(corpus,num_topics=size,eta="auto",workers=6)
    lda.save(model_path)
    return lda

def mini_lda_model(collection,num_topics=10):
    corpus = custom_corpus(collection)
    lda = models.ldamulticore.LdaMulticore(corpus,num_topics=num_topics,eta="auto",workers=6)
    return lda,corpus.dictionary

def google_news_model(model_target_path="model/google_news_model.model",model_source_path="/home/amir/makmal/ular_makan_surat_khabar/word_embeddings/GoogleNews-vectors-negative300.bin"):
    doc2vec = models.doc2vec.Doc2Vec(min_count=10)
    doc2vec.load_word2vec_format(model_source_path,binary=True)
    # doc2vec.init_sims(replace=False)
    doc2vec.save(model_target_path)
    return doc2vec

def save_lda_topics_to_db(lda_model_path,dictionary):
    lda = models.ldamulticore.LdaMulticore.load(lda_model_path)
    i = 1
    for document in collection.find():
        if i%100 == 0:
            print(i)
        i += 1
        document_id = document["_id"]
        cleaned = clean(document["content"])
        doc_bow = dictionary.doc2bow(cleaned.split())
        modified = document
        modified["lda_topics"] = lda[doc_bow]
        del(modified["_id"])
        collection.replace_one({"_id":document_id},modified)
        # topic_ids = []
        # topic_probabilities = []
        # for topic_id,topic_probability in lda[doc_bow]:
        #     topic_ids.append(topic_id)
        #     topic_probabilities.append(topic_probability)

def compute_complete_lda_topics(model_path,size=100):
    corpus = CompleteCorpus()
    corpora.MmCorpus.serialize("corpus/temp_complete_corpus.mm", corpus)
    corpus.dictionary.save("dictionary/temp_complete_dictionary.dict")
    dictionary = corpora.dictionary.Dictionary.load("dictionary/temp_complete_dictionary.dict")
    c = corpora.mmcorpus.MmCorpus("corpus/temp_complete_corpus.mm")
    lda = update_lda_model(c,model_path,size)
    save_lda_topics_to_db(model_path,dictionary)
    return lda, corpus.dictionary

if __name__ == "__main__":
    # for el in quranic_sentences_with_docs():
    #     print(el)
    # print("doc2vec 400")
    # update_doc2vec_model()
    # quran_dictionary = corpora.dictionary.Dictionary.load("dictionary/quranic_ayat_en.dict")


    # LDA Topic Modelling

    corpus = CompleteCorpus()
    # update_lda_model(corpus)

    save_lda_topics_to_db("model/lda.model",corpus.dictionary)

    # Doc2Vec Modelling
    #
    # print("doc2vec modelling")
