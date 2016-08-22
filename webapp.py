from flask import Flask,render_template,url_for,redirect
app = Flask(__name__)
from pymongo import MongoClient
from bson.objectid import ObjectId
from pathlib import Path

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

def get_item(item_id):
    # Convert from string to ObjectId:
    return collection.find_one({'_id': ObjectId(item_id)})

@app.route('/')
def index():
    return redirect(url_for("items_index",page=1))

@app.route('/items')
def items():
    return redirect(url_for("items_index",page=1))

@app.route('/items/page/<page>')
def items_index(page):
    segmentation = 5
    page = int(page)
    skip = (page-1)*segmentation
    limit = segmentation

    items = [el for el in collection.find({ "related_news" : { "$exists" : True } }).sort([["year",-1],["month",-1],["day",-1],["time_string",-1]]).skip(skip).limit(limit)]
    max_page = collection.find({ "related_news" : { "$exists" : True } }).count()/segmentation
    max_page = int(max_page)

    # items = collection.find().skip(skip).limit(limit)
    # print(d)
    modified_items = items

    for idx,item in enumerate(items):
        title_link = []
        for news in item["related_news"]:
            related = collection.find_one({"_id":news})
            title_link.append({"title":related["title"],"link":related["link"]})

        modified_items[idx]["title_link"] = title_link

    return render_template("index.html",items=modified_items,current_page=page,max_page=max_page)

@app.route('/items/<item_id>')
def items_show(item_id):

    item = get_item(item_id)
    verses = [ quranic_lines[i] for i in item["related_verses"]]
    return render_template("show.html",verses=verses,title=item["title"],link=item["link"])
    # except KeyError as e:
    #     return None
