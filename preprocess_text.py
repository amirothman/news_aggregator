from pathlib import Path
import re
from gensim import corpora,models
import numpy as np

def clean(sentence):
    removed_chars = re.sub(r"[^a-zA-Z]"," ",sentence)
    stripped = removed_chars.strip()
    lowered = stripped.lower()
    return lowered

def text_from_file(file_path):
    p = Path(file_path)
    return clean(p.read_text())

def text_from_directory(directory_path):
    p = Path(directory_path)
    return [clean(_file.read_text()) for _file in p.glob("*") if _file.is_file()]

def text_from_json(file_path):
    pass

def text_from_url(url):
    pass

if __name__ == '__main__':
    txt = text_from_file("textfiles/sample1.txt")
    lda = models.ldamulticore.LdaMulticore.load("model/quranic_ayat_en_lda_200.model")
    dictionary = gensim.corpora.dictionary.Dictionary.load("dictionary/quranic_ayat_en.dict")
