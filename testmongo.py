from pymongo import MongoClient

client = MongoClient()
db = client['crawled_news']
collection = db['crawled_news']

page = 4s
skip = (page-1)*20
limit = page*20
items = collection.find().skip(skip).limit(limit)

for item in items:
    print(item)
