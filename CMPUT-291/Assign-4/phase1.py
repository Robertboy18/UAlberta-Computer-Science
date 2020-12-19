import json
from pymongo import MongoClient
import re
import time

port = input("Enter the port number: ")
client = MongoClient('localhost', int(port))
mydb = client['291db']

posts_ = mydb['posts']
tags = mydb['tags']
votes = mydb['votes']


def remove(text):
    """
    Clean the text by removing html tags
    :param text: the text to clean
    :return: the cleaned text
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).lower()


def insert_data():
    """
    Insert all the data from json files to a database and extract terms and add it to posts
    :return: None
    """
    collist = mydb.list_collection_names()
    new = []
    for word in collist:
        new.append(word.lower())
    if "posts" in new:
        print("The posts collection exists. Dropping it")
        posts_.drop()
    if "votes" in new:
        print("The votes collection exists. Dropping it")
        tags.drop()
    if "tags" in new:
        print("The tags collection exists. Dropping it")
        votes.drop()

    with open("Posts.json") as f:
        file_data = json.load(f)["posts"]["row"]
        counter = 0
        for document in file_data:
            terms1, terms2 = [], []
            if "Body" in document:
                Body_string = remove(document["Body"]).split()
                terms1 = [word for word in Body_string if len(word) > 2]
            if "Title" in document:
                Title_string = remove(document["Title"]).split()
                terms2 = [word for word in Title_string if len(word) > 2]
            file_data[counter]["terms"] = list(set(terms1 + terms2))
            counter += 1
        posts_.insert_many(file_data)
        posts_.create_index("terms")

    with open("Tags.json") as f:
        file_data = json.load(f)  # load data from JSON to dict
        tags.insert_many(file_data["tags"]["row"])

    with open("Votes.json") as f:
        file_data = json.load(f)  # load data from JSON to dict
        votes.insert_many(file_data["votes"]["row"])

start_time = time.time()
insert_data()
print("--- %s seconds ---" % (time.time() - start_time))