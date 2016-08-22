import requests
from readability.readability import Document
from bs4 import BeautifulSoup

def extract_from_url(link):
    r  = requests.get(link)
    html = r.text
    readable = Document(html).summary()
    cleaned = BeautifulSoup(readable,"lxml").text

    return cleaned
