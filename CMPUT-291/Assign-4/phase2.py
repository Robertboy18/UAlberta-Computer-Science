import random
import sys
from datetime import datetime

from pymongo import MongoClient

port = input("Enter the port number: ")
client = MongoClient('localhost', int(port))
db = client['291db']
post_table = db["posts"]
vote_table = db["votes"]
tag_table = db["tags"]

random.seed(1000000)

currentDate = datetime.strftime(datetime.today(), "%Y-%m-%dT%H:%M:%S.000")


def menu(uid):
    """
    It calls the respective task functions when a task is selected
    :param uid: userid of the user performing the task
    :return: None
    """
    variable = True
    while variable:
        print("""
                \t \t \t Select an option to perform the following task - 
                \t \t \t 1 - Post a question
                \t \t \t 2 - Search for questions
                \t \t \t 3 - Question action-Answer
                \t \t \t 4 - Question action- List Answers
                \t \t \t 5 - Post action-Vote
                \t \t \t 6 - Exit
                """)
        try:
            input_user = int(input())
        except:
            print("Invalid Input")
            continue
        if input_user == 1:
            post(uid)
        elif input_user == 2:
            search()
        elif input_user == 3:
            pid_selected = search()
            answer(uid, pid_selected)
        elif input_user == 4:
            pid_selected = search()
            list_answers(uid, pid_selected)
        elif input_user == 5:
            pid_selected = search()
            vote(uid, pid_selected)
        elif input_user == 6:
            print("Goodbye")
            sys.exit()
        else:
            print("Invalid Input")
            continue


def report(uid):
    """
    Calls the functions for each report
    :param uid: userid of the user performing the task
    :return: None
    """
    print("User: " + str(uid))
    user_posts_dets(uid)
    user_posts_dets(uid, 2)
    user_votes_dets(uid)


def user_votes_dets(uid):
    """
    Gets the number of votes associated with the user and prints it
    :param uid: userid of the user performing the task
    :return: None
    """
    user = vote_table.find({"UserId": str(uid)})
    count = 0
    for x in user:
        count += 1
    print(" \t \t Number of votes: " + str(count))


def user_posts_dets(uid, kind=1):
    """
    Gets the post details and prints it
    :param uid: userid of the user performing the task
    :param kind: if kind = 1 means get details for questions
    :return: None
    """
    total = post_table.aggregate([{"$match": {"$and": [{"PostTypeId": str(kind)}, {"OwnerUserId": str(uid)}]}},
                                  {"$group": {"_id": {"OwnerUserId": "$OwnerUserId"}, "count": {"$sum": 1},
                                              "average": {"$avg": "$Score"}}
                                   }])

    for i in total:
        if kind == 1:
            print(" \t \t Posted: " + str(i["count"]) + " question/s \n \t \t Average score: " + str(i["average"]))
        else:
            print(" \t \t Posted: " + str(i["count"]) + " answer/s \n \t \t Average score: " + str(i["average"]))


def login():
    """
    Intial login screen which asks the user if they are providing a userid or not and calls report and menu accordingly
    :return: None
    """
    option = input("Do you want to provide a user id?(y/n)")
    if option.lower()[0] == "y":
        userid = input("Please enter your user id: ")
        report(userid)
        menu(userid)
    else:
        menu("NA")


def post(uid):
    """
    Performs post action with all the details the user provides
    :param uid: userid of the user performing the task
    :return: None
    """
    title_q = input("Please enter the title of the Post : ")
    title_b = input("Please enter the body of the Post : ")
    tags = input("Please enter the tags : ").split()
    tag_posts = tag_table.distinct("TagName")
    for j in range(len(tags)):
        if tags[j] in tag_posts:
            k = tag_table.find({"TagName": tags[j]})
            r = 0
            l = 0
            for i in k:
                l = i["Count"]
                r = l + 1
                finder = {'TagName': tags[j]}
                new_value = {"$set": {"Count": r}}
                tag_table.update_one(finder, new_value)
        else:
            unique_tag_id = str(random.randint(100, 23810381))
            already_tags = tag_table.find({"Id": unique_tag_id})
            while len(list(already_tags)) > 0:
                unique_tag_id = str(random.randint(200, 9999999))
            new = {"Id": unique_tag_id, "Count": 1}
            tag_table.insert_one(new)

    unique_id = str(random.randint(200, 9999999))
    already_there = post_table.find({"Id": unique_id})
    while len(list(already_there)) > 0:
        unique_id = str(random.randint(200, 9999999))

    if uid != "NA":
        doc1 = {"Id": unique_id, "PostTypeId": 1, "OwnerUserId": uid, "Title": title_q, "Body": title_b, "Tags": tags,
                "Date": currentDate,
                "ViewCount": 0,
                "AnswerCount": 0, "CommentCount": 0, "FavoriteCount": 0, "c License": "CC BY-SA 2.5"}
        post_table.insert_one(doc1)
    else:
        doc1 = {"Id": unique_id, "PostTypeId": 1, "Title": title_q, "Body": title_b, "Tags": tags,
                "Date": currentDate,
                "ViewCount": 0,
                "AnswerCount": 0, "CommentCount": 0, "FavoriteCount": 0, "c License": "CC BY-SA 2.5"}
        post_table.insert_one(doc1)


def answervote(uid):
    print("Done")


def list_answers(uid, pid):
    """
    Lists all the answers associated to the question selected
    :param uid: userid of the user performing the task
    :param pid: pid of the post
    :return:
    """
    accepted = post_table.find({"Id": str(pid), "PostTypeId": "1"})
    q = 0
    z = []
    check = []
    k = 1
    g = []
    for i in accepted:
        if "AcceptedAnswerId" in i:
            q = i["AcceptedAnswerId"]
            g.append(i["AcceptedAnswerId"])
        else:
            print("No Accepted answer ")
    already_there = post_table.find({"ParentId": str(pid), "PostTypeId": "2"})
    already = post_table.find({"Id": q, "ParentId": str(pid), "PostTypeId": "2"})

    print("#  |", "", " Body ", " |   Date ", "  | Score |")
    if q != 0:
        for i in already:
            print(str(k), " * ", i["Body"][:81], " ", i["CreationDate"], " ", i["Score"])
            z.append(k)
            check.append(i["Id"])
            k += 1

    else:
        check.append(-1)

    for i in already_there:
        if i["Id"] != check[0]:
            print(str(k) + "    ", i["Body"][:81], " ", i["CreationDate"], " ", i["Score"])
            z.append(k)
            g.append(i["Id"])
            k += 1

    r = int(input("Please enter your answer choice to view the entire details : "))
    if r not in z:
        print("Invalid Input ")
        while r not in z:
            r = int(input("Please enter your answer choice to view the entire details : "))
    else:
        final = post_table.find({"Id": g[r - 1], "ParentId": str(pid), "PostTypeId": "2"})
        final_id = 0
        for i in final:
            print("Id : ", i["Id"])
            final_id = i["Id"]
            print("PostTypeId : ", i["PostTypeId"])
            print("ParentId : ", i["ParentId"])
            print("OwnerUserId : ", i["OwnerUserId"])
            print("Body : ", i["Body"])
            print("Date : ", i["CreationDate"])
            print("Score : ", i["Score"])
            print("CommentCount : ", i["CommentCount"])
            print("c License : ", i["cLicense"])
    dec = input("Do you want to vote [y/n] : ")
    if dec[0] == "y":
        print("Voting")
        vote(uid, pid)
    else:
        pass


def answer(uid, pid):
    """
    Performs post action answer on questions and checks if userid was provided and adds the field userid otherwise does not
    :param uid: userid of the user performing the task
    :param pid: pid of the question
    :return: None
    """
    title_b = input("Please enter the body of the Post : ")
    unique_id = str(random.randint(200, 9999999))
    already_there = post_table.find({"Id": unique_id})
    while len(list(already_there)) > 0:
        unique_id = str(random.randint(200, 9999999))
    if uid != "NA":
        doc1 = {"Id": unique_id, "PostTypeId": 2, "ParentId": pid, "OwnerUserId": uid, "Body": title_b,
                "Date": currentDate,
                "Score": 0, "CommentCount": 0, "c License": "CC BY-SA 2.5"}
        post_table.insert_one(doc1)
    else:
        doc1 = {"Id": unique_id, "PostTypeId": 2, "ParentId": pid, "Body": title_b, "Date": currentDate,
                "Score": 0, "CommentCount": 0, "Content License": "CC BY-SA 2.5"}
        post_table.insert_one(doc1)


def vote(userid, pid):
    """
    Performs voting and checks if userid was provided and adds the field userid otherwise does not
    :param userid: userid of the user performing the task
    :param pid: pid of post
    :return: None
    """
    unique_id = str(random.randint(200, 9999999))
    already_there = vote_table.find({"Id": unique_id})
    while len(list(already_there)) > 0:
        unique_id = str(random.randint(200, 9999999))
    if str(userid) != "NA":
        user = vote_table.find({"$and": [{"UserId": userid}, {"PostId": pid}]})
        if len(list(user)) > 0:
            print("You have already voted on this post")
        else:
            doc = {
                "Id": unique_id,
                "PostId": str(pid),
                "VoteTypeId": 2,
                "UserId": userid,
                "CreationDate": currentDate
            }
            vote_table.insert_one(doc)
            post_table.update_one({"Id": pid}, {"$inc": {"Score": 1}})
            print("Voting")
    else:
        vote_table.insert_one({
            "Id": unique_id,
            "PostId": pid,
            "VoteTypeId": 2,
            "CreationDate": currentDate
        })
        post_table.update_one({"Id": pid}, {"$inc": {"Score": 1}})
        print("Voting")


def search(kind=1):
    """
    Searches for the posts based on keywords provided and calls search helper function
    :param kind: if kind is 1 then only questions are returned
    :return: None
    """
    words = input("Enter some keywords: ").split()
    pids = {}  # {pid: [[matching words], number of votes, number of answers],...}

    # get all valid pids
    for word in words:
        search_helper(word, pids, kind)

    arr = list(pids.keys())

    while arr:
        j = 1
        print("\nChoose one of the following options 1 - 5 from (" + str(
            len(arr)) + " results remain) or press newline to continue")
        while j < 6:
            try:
                chosen = post_table.find_one({"_id": arr[j - 1]})
                print("\t \t" + str(j), end=":")
                if "Title" in chosen.keys():
                    print("\n\t \t \t Title", chosen["Title"], sep=": ", end=" ")
                if "Date" in chosen.keys():
                    print("\n\t \t \t Date", chosen["Date"], sep=": ", end=" ")
                if "Score" in chosen.keys():
                    print("\n\t \t \t Score", chosen["Score"], sep=": ", end=" ")
                if "AnswerCount" in chosen.keys():
                    print("\n\t \t \t AnswerCount", chosen["AnswerCount"], sep=": ", end=" ")
                print("\n")
                j += 1
            except:
                print("That's it")
                break
        choice = input()
        if choice == "":
            for x in range(5):
                try:
                    arr.pop(0)
                except:
                    print("That was all returning to menu!")
                    return
            continue
        try:
            if int(choice) in [1, 2, 3, 4, 5]:
                # if we've chosen a post
                chosen = post_table.find_one({"_id": arr[int(choice) - 1]})
                print("More info on the chosen document:")
                for key in chosen:
                    print("\n\t \t \t " + key, chosen[key], sep=": ", end=" ")
                new_count = chosen["ViewCount"] + 1

                finder = {'_id': arr[int(choice) - 1]}
                new_value = {"$set": {"ViewCount": new_count}}
                post_table.update_one(finder, new_value)
                return int(chosen["Id"])
            else:
                print("Invalid input")
        except:
            print("Invalid input")
    print("There is nothing with that so returning to menu")


def search_helper(word, running_pids, kind=1):
    """
    Given a word search in tags, title and body
    :param word (string)
    :param running_pids (dictionary)
    """

    if kind == 1:
        title_posts = post_table.find({"$and": [{"PostTypeId": "1"}, {"Title": {"$regex": word, "$options": 'i'}}]})
        body_posts = post_table.find({"$and": [{"PostTypeId": "1"}, {"Body": {"$regex": word, "$options": 'i'}}]})
        tag_posts = post_table.find({"$and": [{"PostTypeId": "1"}, {"Tags": {"$regex": word, "$options": 'i'}}]})
    else:
        title_posts = post_table.find({"Title": {"$regex": word, "$options": 'i'}})
        body_posts = post_table.find({"Body": {"$regex": word, "$options": 'i'}})
        tag_posts = post_table.find({"Tags": {"$regex": word, "$options": 'i'}})
    for post in title_posts:
        actual = post['_id']  # the id of our thing
        if actual in running_pids.keys() and word not in running_pids[actual][0]:
            running_pids[actual][0].append(word)
            # if we have that the pid is not in our running pids
        elif actual not in running_pids.keys():
            running_pids[actual] = [[word]]
        # it is in our pids and for this word
        else:
            pass

    # search in body
    for post in body_posts:
        actual = post['_id']  # NB tuples
        # if we have that pid is in our running pids but not for this word
        if actual in running_pids.keys() and word not in running_pids[actual][0]:
            running_pids[actual][0].append(word)
        # if we have that the pid is not in our running pids
        elif actual not in running_pids.keys():
            running_pids[actual] = [[word]]
        # it is in our pids and for this word
        else:
            pass

    # search in tags
    for post in tag_posts:
        actual = post['_id']  # NB tuples
        # if we have that pid is in our running pids but not for this word
        if actual in running_pids.keys() and word not in running_pids[actual][0]:
            running_pids[actual][0].append(word)
        # if we have that the pid is not in our running pids
        elif actual not in running_pids.keys():
            running_pids[actual] = [[word]]
        # it is in our pids and for this word
        else:
            pass


login()
