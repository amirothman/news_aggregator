import subprocess
from pymongo import MongoClient
from pathlib import Path
import numpy as np

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']


def train_fast_text():
    with open("textfiles/tempfile_input.txt","w") as f:
        for t in collection.find():
            f.write(t["content"])
            f.write("\n")

    subprocess.run(["./fasttext","skipgram","-input","textfiles/tempfile_input.txt","-output","model/fast_text"])

def query_fast_text(text):
    subprocess.run(["sh","fast_text_vector.sh",text])
    p = Path("textfiles/fast_text_word_vectors.txt")
    vector = [ [float(n) for n in line.split()[1:]] for line in p.read_text().split("\n")[:-1]]
    v = np.array(vector)
    return np.sum(v,axis=0)
    
if __name__ == '__main__':
    # train_fast_text()
    summed = np.zeros((0,100))
    v = query_fast_text("minister dying now \asd k alsdlkalsndkas\ ds")
    summed = np.sum(v,axis=0)
    print(summed)
