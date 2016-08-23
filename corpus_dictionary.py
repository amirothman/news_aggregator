from gensim import corpora
from pymongo import MongoClient
from preprocess_text import clean

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

class CustomCorpus(object):
    def __init__(self,query={}):
        self.query = query
        self.dictionary = corpora.Dictionary(all_of_words(query=query))
        self.dictionary.filter_extremes(no_below=1, keep_n=30000) # check API docs for pruning params

    def __iter__(self):
        for tokens in all_of_words(query=self.query):
            yield self.dictionary.doc2bow(tokens)

class SubCorpus(object):
    def __init__(self,collection):
        self.collection = collection
        self.dictionary = corpora.Dictionary([clean(d["content"]).split() for d in collection])
        self.dictionary.filter_extremes(no_below=1, keep_n=30000) # check API docs for pruning params

    def __iter__(self):
        for l in self.collection:
            yield self.dictionary.doc2bow(clean(l["content"]).split())


def iterate_collection(collection):
    for doc in collection:
        cleaned = clean(doc["content"])
        yield cleaned.split()

def all_of_words(query={}):
    for doc in collection.find(query):
        yield doc["content"].split()

def recreate_dictionary(dict_path = 'dictionary/all_of_words.dict', corpus_path = 'corpus/all_of_words.mm'):
    corpus = CustomCorpus(query) # create a dictionary
    corpus.dictionary.save(dict_path)
    corpora.MmCorpus.serialize(corpus_path, corpus)

def custom_corpus(collection):
    corpus = SubCorpus(collection)
    # corpora.MmCorpus.serialize("corpus/temp.mm", corpus)
    # c = corpora.mmcorpus.MmCorpus("corpus/temp.mm")
    return corpus


if __name__ == "__main__":
    corpus = CustomCorpus() # create a dictionary
    # corpus.dictionary
    # corpus.dictionary.save('dictionary/all_of_words.dict')
    # corpora.MmCorpus.serialize('corpus/all_of_words.mm', corpus)
