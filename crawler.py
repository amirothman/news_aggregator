import feedparser
import json
import requests
from dateutil.parser import parse
from readability.readability import Document
from bs4 import BeautifulSoup
import re
import os
from pymongo import MongoClient
from preprocess_text import clean
from datetime import datetime
from pathlib import Path
from similarity import fast_text_vector_from_redis
import redis
from annoy import AnnoyIndex
import random

def extract_content(link):
    r  = requests.get(link)
    html = r.text
    readable = Document(html).summary()
    return BeautifulSoup(readable,"lxml").text


def crawl():
    client = MongoClient()
    db = client['crawled_news']
    collection = db['crawled_news']
    r = redis.StrictRedis(host='localhost', port=6379, db=0)

    new_docs = 0
    rss_feeds = []
    with open("news_rss.txt","r") as news_rss:
        for line in news_rss:
            re.sub("\n","",line)
            rss_feeds.append(line)
            
    random.shuffle(rss_feeds)

    path_to_index = "similarity_index/fast_text"
    p = Path(path_to_index)
    if p.is_file():
        u = AnnoyIndex(100)
        u.load(path_to_index)



    for feed in rss_feeds:
        parsed_feed = feedparser.parse(feed)
        try:
            links_dates = [ (el["link"],el["published"],el["title"]) for el in parsed_feed.entries]

            for link,date,title in links_dates:
                if not collection.find_one({"link":link}):
                    print(link)
                    date_parsed = parse(date)
                    content = extract_content(link)
                    file_name_to_save = re.sub(r"\W","_",link)

                    # summarized = summarize(content, word_count=50)
                    n = datetime.now()
                    time_string = "{}{}{}{}{}{}{}".format(date_parsed.year,date_parsed.month,date_parsed.day,n.hour,n.minute,n.second,n.microsecond)
                    article_dict = {"title":title,
                                    "content":content,
                                    "link":link,
                                    "day":date_parsed.day,
                                    "month":date_parsed.month,
                                    "year":date_parsed.year,
                                    "time_string":time_string
                                    # "summarize":summarized
                                    }

                    json_string = json.dumps(article_dict, sort_keys=True, indent=4)
                    if p.is_file():
                        vector = fast_text_vector_from_redis(content,r)
                        sim_idx = u.get_nns_by_vector(vector,10)
                        # sim_idx = list(set(sim_idx))
                        nearest_neighbour = [collection.find_one({"integer_id":idx})for idx in sim_idx]

                        nearest_neighbour_ids = [n["_id"] for n in nearest_neighbour if n]

                        article_dict["related_news_fast_text"] = nearest_neighbour_ids
                        article_dict["object_id_by_divergence"] = nearest_neighbour_ids
                    print(json_string)
                    collection.insert_one(article_dict)
                    new_docs += 1
                else:
                    print(link,' already saved')
                # with(open("collection/{0}.json".format(file_name_to_save),"w")) as f:
                #     f.write(json_string)
        except KeyError:
            print("KeyError")
        except requests.exceptions.ConnectionError:
            print("requests.exceptions.ConnectionError")
        except requests.exceptions.ChunkedEncodingError:
            print("requests.exceptions.ChunkedEncodingError")

    return new_docs

if __name__=="__main__":
    while True:
        crawl()
