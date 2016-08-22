import feedparser
import json
import requests
from dateutil.parser import parse
from readability.readability import Document
from bs4 import BeautifulSoup
import re
import os
from gensim.summarization import summarize
from pymongo import MongoClient
from preprocess_text import clean
from datetime import datetime

def extract_content(link):
    r  = requests.get(link)
    html = r.text
    readable = Document(html).summary()
    cleaned = BeautifulSoup(readable,"lxml").text
    c = clean(cleaned)
    return c


def crawl():
    client = MongoClient()
    db = client['crawled_news']
    collection = db['crawled_news']
    # The more merrier. Using four just as example.
    new_docs = 0
    rss_feeds = [

                # International news
                "http://feeds.reuters.com/Reuters/PoliticsNews",
                "http://feeds.reuters.com/reuters/topNews",
                "http://feeds.reuters.com/Reuters/worldNews",
                "http://feeds.reuters.com/Reuters/domesticNews",
                "http://feeds.bbci.co.uk/news/world/rss.xml",
                "http://feeds.bbci.co.uk/news/politics/rss.xml",
                "http://feeds.bbci.co.uk/news/uk/rss.xml",
                "http://www.spiegel.de/international/index.rss",
                #"http://www.spiegel.de/international/world/index.rss",
                #"http://www.spiegel.de/international/europe/index.rss",
                #"http://rss.cnn.com/rss/edition.rss",
                "http://rss.cnn.com/rss/edition_world.rss",
                "http://www.usnews.com/rss/news",
                "http://rss.upi.com/news/tn_int.rss",
                "http://rss.upi.com/news/tn_us.rss",
                "http://rss.upi.com/news/top_news.rss",
                "http://www.france24.com/en/top-stories/rss",
                "http://www.france24.com/en/france/rss",
                "http://www.france24.com/en/europe/rss",
                "http://www.nytimes.com/services/xml/rss/nyt/World.xml",
                "http://www.nytimes.com/services/xml/rss/nyt/US.xml",
                "http://gulfnews.com/cmlink/1.446094",
                "http://hosted2.ap.org/atom/APDEFAULT/cae69a7523db45408eeb2b3a98c0c9c5",
                "http://hosted2.ap.org/atom/APDEFAULT/89ae8247abe8493fae24405546e9a1aa",
                "http://hosted2.ap.org/atom/APDEFAULT/3d281c11a96b4ad082fe88aa0db04305",
                "http://apps.shareholder.com/rss/rss.aspx?channels=7142&companyid=ABEA-4M7DG8&sh_auth=2654998770%2E0%2E0%2E42600%2E95aa53563651b169e9250a1144c817be",
                "http://www.ibtimes.co.uk/rss/world",
                "http://www.astm.org/RSS/NS.rss"

                # Malaysian news
                # "http://www.agendadaily.com/index.php?format=feed&type=rss",
                 "http://www.malaysia-today.net/feed/",
                 "http://www.newsplus.my/feed/",
                 "http://news.google.com/news?hl=en&gl=us&q=malaysiakini&um=1&ie=UTF-8&output=rss",
                 "http://www.therakyatpost.com/category/news/feed/",
                #  "http://feeds.feedburner.com/malaysiandigest/Xrpu",
                 "http://malaysia-chronicle.com/index.php?option=com_k2&view=itemlist&format=feed&type=rss&Itemid=2",
                 "http://www.thesundaily.my/rss",
                 "http://themalaysianreserve.com/new/rss.xml",
                 "https://www.freemalaysiatoday.com/category/nation/feed",
                 "http://www.themalaymailonline.com/feed/rss/malaysia",
                #  "http://www.harakahdaily.net/index.php?format=feed&type=rss",
                 "http://www.nst.com.my/latest.xml",
                 "http://www.thestar.com.my/rss/news/nation/",
                 "http://english.astroawani.com/rss/national/public"
                 ]

    for feed in rss_feeds:
        parsed_feed = feedparser.parse(feed)

        links_dates = [ (el["link"],el["published"],el["title"]) for el in parsed_feed.entries]

        for link,date,title in links_dates:
            if not collection.find_one({"link":link}):
                date_parsed = parse(date)
                content = extract_content(link)
                file_name_to_save = re.sub(r"\W","_",link)

                # summarized = summarize(content, word_count=50)
                n = datetime.now()
                time_string = "{}{}{}{}{}{}{}".format(n.year,n.month,n.day,n.hour,n.minute,n.second,n.microsecond)
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


                print(json_string)
                collection.insert_one(article_dict)
                new_docs += 1
            else:
                print(link,' already saved')
            # with(open("collection/{0}.json".format(file_name_to_save),"w")) as f:
            #     f.write(json_string)

    return new_docs

if __name__=="__main__":
    crawl()
