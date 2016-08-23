from gensim import corpora
from pymongo import MongoClient

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

def all_of_words():
    for doc in collection.find():
        yield doc["content"].split()

class CustomCorpus(object):
    def __init__(self):
        self.dictionary = corpora.Dictionary(all_of_words())
        self.dictionary.filter_extremes(no_below=1, keep_n=30000) # check API docs for pruning params

    def __iter__(self):
        for tokens in all_of_words():
            yield self.dictionary.doc2bow(tokens)

def recreate_dictionary(dict_path = 'dictionary/all_of_words.dict', corpus_path = 'corpus/all_of_words.mm'):
    corpus = CustomCorpus() # create a dictionary
    corpus.dictionary.save(dict_path)
    corpora.MmCorpus.serialize(corpus_path, corpus)

if __name__ == "__main__":
    corpus = CustomCorpus() # create a dictionary
    corpus.dictionary.save('dictionary/all_of_words.dict')
    corpora.MmCorpus.serialize('corpus/all_of_words.mm', corpus)
