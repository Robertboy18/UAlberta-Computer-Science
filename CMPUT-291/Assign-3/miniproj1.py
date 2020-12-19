import sqlite3
import time
import getpass
import sys
from datetime import datetime
import random as r
import getopt


def main(argv):
    """
    checks for the file name entered by the user as a command line argument using -i
    opts is the list of options that the program will recognize which is -i
    :param argv:
    :return: inputfile - the filename entered by the user
    """
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv[0:], "hi:o:", ["ifile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> ')
        sys.exit(2)
    if len(argv) < 2:
        print('test.py -i <inputfile> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg

    return inputfile


# get the filename and store it where sys.arg gives the list of command line arguments passed
inputfile = main(sys.argv[1:])
conn = sqlite3.connect(inputfile)  # connect to the database filename
c = conn.cursor()


def mark(uid, pid):
    """
    Mark an answer as an accepted answer by updating the theaid in questions table
    :param uid: the userid of the user marking the answer
    :param pid: the pid of the answer being marked
    :return: None
    """
    # get the qid from answers where the pid is equal to that provided
    c.execute("SELECT DISTINCT qid FROM answers where pid = ?", (str(pid),))
    qid = c.fetchone()
    conn.commit()

    # get the theaid from questions
    c.execute("SELECT DISTINCT theaid FROM questions where pid = ?", (qid[0],))
    theaid = c.fetchone()
    conn.commit()
    fill = (str(pid), qid[0])

    # if there is already an accepted answer then ask the user if they want to change it
    if theaid[0] != None:
        print("The question has already an accepted answer.")
        change = input("Do you want to change the accepted answer(Y,N) ? ")
        if change[0].lower() == "n":
            filter(uid)
        else:
            c.execute("Update questions set theaid = ? where pid = ?", fill)
            conn.commit()
            print("Successfully marked answer {} as accepted answer for question {}".format(pid, qid[0]))
    else:
        c.execute("Update questions set theaid = ? where pid = ?", fill)
        conn.commit()
        print("Successfully marked answer {} as accepted answer for question {}".format(pid, qid[0]))


def tag(uid, pid):
    """
    Insert the tag into the tags table and check if that tag wasn't already assigned to that post
    :param uid: the userid of the user adding the tag
    :param pid: the pid of the post in which the tag is added
    :return: None
    """
    tag_input = input("Please enter the tag: ").lower()
    c.execute("SELECT DISTINCT tag from tags where pid =?;", (pid,))
    invalid_tags = c.fetchall()
    for i in invalid_tags:
        i[0].lower()

    # if the tag is already not there then assign a tag to that post
    if (tag_input,) not in invalid_tags:
        tag_insert = "insert into tags values(?,?)"
        c.execute(tag_insert, (pid, tag_input))
        conn.commit()
        print("Successfully added tag {} to post {}".format(tag_input, pid))
    else:
        print("Post {} already has tag {}".format(pid, tag_input))
        filter(uid)


def badge(uid, pid):
    """
    Insert the badge into the ubadges table and check if the badgename is a valid one
    by checking badge names in badges table
    :param uid: uid of user giving the badge
    :param pid: the pid of the post in which the badge is added
    :return: None
    """
    # get valid badgenames from badges table
    c.execute("SELECT DISTINCT bname from badges;")
    valid_bnames = c.fetchall()
    currentDate = datetime.today().strftime("%Y-%m-%d")
    badge_name = input("Please enter the badge name ")
    c.execute("SELECT poster from posts WHERE pid = ?", (pid,))
    poster = c.fetchone()[0]

    # add the badge if it is a valid badge name otherwise send the user back to the tasks screen
    if (badge_name,):# in valid_bnames:
        badge_insert = "insert into ubadges values(?,?,?)"
        c.execute(badge_insert, (str(poster), currentDate, badge_name))
        conn.commit()
        print("Successfully assigned badge to user {}".format(poster))
    else:
        print("Not a valid badge name.")
        filter(uid)


def edit(pid):
    """
    Ask the user if they want to edit the title and/or the body and make respective edits by
    updating the posts table
    :param pid: the pid of the post which is being edited
    :return: None
    """
    title_1 = input("Do you want to edit the title of the post y/n ")
    if title_1 == 'y':
        title_q = input("Please enter the new title of the Post : ")
        r = [(title_q, pid)]
        c.executemany("Update posts set title = ? where pid = ?", r)
        conn.commit()
        title_2 = input("Do you want to edit the body of the post y/n ")
        if title_2 == 'y':
            title_b = input("Please enter the new body of the Post : ")
            k = [(title_b, pid)]
            c.executemany("Update posts set body = ? where pid = ?", k)
            conn.commit()
        else:
            print("No edits done")
    else:
        title_2 = input("Do you want to edit the body of the post y/n ")
        if title_2 == 'y':
            title_b = input("Please enter the new title of the Post : ")
            k = [(title_b, pid)]
            c.executemany("Update posts set body = ? where pid = ?", k)
            conn.commit()
        else:
            print("No edits done ")


def post(uid):
    """
    Make a post and ensure a unique pid is assigned to the post by checking if the pid is not there in the posts table
    :param uid: userid of the user making the post
    :return: None
    """
    # providing title and body texts
    title_q = input("Please enter the title of the Post : ")
    title_b = input("Please enter the body of the Post : ")
    currentDate = datetime.today().strftime("%Y-%m-%d")
    pid = "p0" + str(r.randint(100, 999))

    # get all the pids
    c.execute("SELECT DISTINCT pid FROM posts")
    pids = c.fetchall()

    # if pid is already taken the reassign
    while pid in pids:
        pid = "p0" + str(r.randint(100, 999))

    # insert the post into posts table
    data = [(pid, currentDate, title_q, title_b, uid)]
    insertData = '''INSERT INTO posts(pid, pdate, title, body, poster)
        VALUES(?,?,?,?,?)'''
    c.executemany(insertData, data)
    conn.commit()

    # insert the post into questions table and initially provide None for theaid
    insert_question = '''INSERT INTO questions(pid, theaid) VALUES(?,?)'''
    q_data = [(pid, None)]
    c.executemany(insert_question, q_data)
    conn.commit()

    # closing message
    print("Successfully posted by {} at {}".format(uid, pid))


def search():
    """
    Search for posts.The user should be able to provide one or more keywords,
    and the system should retrieve all posts that contain at least one keyword either
    in title, body, or tag fields. For each matching post, in addition to the columns of posts table,
    the number of votes, and the number of answers if the post is a question (or zero if the question has no answers)
    should be displayed. The result should be ordered based on the number of matching keywords with posts matching the largest
    number of keywords listed on top. If there are more than 5 matching posts, at most 5 matches will be shown at a time,
     letting the user select a post or see more matches. The user should be able
     to select a post
     and perform a post action (as discussed next).
     :param : None
     :return: Selected pid
     """
    words = input("Enter some keywords: ").split()
    pids = {}  # {pid: [[matching words], number of votes, number of answers],...}
    final_pids = []  # [[pid, number of matching words, number of votes, number of answers]...]

    # get all valid pids
    for word in words:
        search_helper(word, pids)

    c.execute("SELECT pid FROM questions;")
    questions = c.fetchall()
    # get the number of votes
    for pid in pids.keys():
        c.execute("SELECT pid, count(*) FROM votes WHERE pid = ? GROUP BY pid;", (pid,))
        avoiding_nullity = c.fetchone()
        if avoiding_nullity is not None:
            no_of_votes = avoiding_nullity[1]
            pids[pid].append(no_of_votes)
        else:
            pids[pid].append(0)

        # (and number of answers if question)
        if (pid,) in questions:
            c.execute("SELECT qid, count(*) FROM answers WHERE qid = ? GROUP BY qid;", (pid,))
            no_of_answers = c.fetchone()
            # if there exists answers
            if no_of_answers is not None:
                pids[pid].append(no_of_answers[1])
            else:
                pids[pid].append(0)
        else:
            pids[pid].append("Non Applicable")

    # moved into final_pids
    for pid in pids.keys():
        final_pids.append([pid, len(pids[pid][0]), pids[pid][1],
                           pids[pid][2]])  # [[pid, number of matching words, number of votes, number of answers]...]

    # order by number of keywords
    final_pids = sorted(final_pids, key=lambda x: x[1], reverse=True)

    # split into each 5 posts (add all post columns)
    print("Choose an option or newline to see more")
    while final_pids:
        for i in range(1, 6):
            try:
                pid = final_pids[i - 1]
                if final_pids:
                    c.execute("SELECT * from posts where pid =?;", (pid[0],))
                    stuff = c.fetchone()
                    print("{}: {}\n".format(i, pid[0]) + "\ttitle: {}\n \tbody: {}\n \tposted on: {} by {}\n".format(
                        stuff[2], stuff[3], stuff[1], stuff[4]) + "\t{} matches ".format(pid[1]) + str(
                        pid[2]) + " votes and " + str(pid[3]) + " answer/s")
                else:
                    pass
            except:
                print("That's it")
                break
        choice = input("")
        if choice == "":
            for i in range(5):
                try:
                    final_pids.pop(0)
                except:
                    print("That was all returning to menu!")
                    break
            continue
        try:
            if int(choice) == 1:
                return final_pids[0][0]
            elif int(choice) == 2:
                return final_pids[1][0]
            elif int(choice) == 3:
                return final_pids[2][0]
            elif int(choice) == 4:
                return final_pids[3][0]
            elif int(choice) == 5:
                return final_pids[4][0]
            else:
                print("Invalid input")
        except:
            print("Invalid input")


def search_helper(word, running_pids):
    """
    Given a word search in tags, title and body
    :param word (string)
    :param running_pids (dictionary)
    """
    # search in title
    word = "%" + word + "%"
    c.execute("SELECT pid FROM posts WHERE title like ?", (word,))
    title_posts = c.fetchall()
    for post in title_posts:
        actual = post[0]  # NB tuples
        # if we have that pid is in our running pids but not for this word
        if actual in running_pids.keys() and word not in running_pids[actual][0]:
            running_pids[actual][0].append(word)
        # if we have that the pid is not in our running pids
        elif actual not in running_pids.keys():
            running_pids[actual] = [[word]]
        # it is in our pids and for this word
        else:
            pass

    # search in body
    c.execute("SELECT pid FROM posts WHERE body like ?", (word,))
    body_posts = c.fetchall()
    for post in body_posts:
        actual = post[0]  # NB tuples
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
    c.execute("SELECT pid FROM tags WHERE tag like ?", (word,))
    tag_posts = c.fetchall()
    for post in tag_posts:
        actual = post[0]  # NB tuples
        # if we have that pid is in our running pids but not for this word
        if actual in running_pids.keys() and word not in running_pids[actual][0]:
            running_pids[actual][0].append(word)
        # if we have that the pid is not in our running pids
        elif actual not in running_pids.keys():
            running_pids[actual] = [[word]]
        # it is in our pids and for this word
        else:
            pass


def answer(uid, qid):
    """
    Add the answer to answers table
    :param uid: userid of the user is posting the answer
    :param qid: the pid of the answer
    :return: None
    """
    # prompt and store entry from the user
    title_q = input("Please enter the title of the Answer : ")
    title_b = input("Please enter the body of the Answer : ")

    # get the date
    currentDate = datetime.today().strftime("%Y-%m-%d")

    # get an unused pid
    pid = "p0" + str(r.randint(100, 999))
    c.execute("SELECT DISTINCT pid FROM posts")
    pids = c.fetchall()
    while pid in pids:
        pid = "p0" + str(r.randint(100, 999))

    # insert the data
    data = [(pid, currentDate, title_q, title_b, uid)]
    insertData = '''INSERT INTO posts(pid, pdate, title, body, poster)
            VALUES(?,?,?,?,?)'''
    c.executemany(insertData, data)
    conn.commit()

    # insert the answers
    insert_ans = '''INSERT INTO answers(pid, qid) VALUES(?,?)'''
    a_data = [(pid, qid)]
    c.executemany(insert_ans, a_data)
    conn.commit()

    # closing message
    print("Successfully posted answer {} for question {}".format(pid, qid))


def vote(uid, pid):
    """
    Inserts the vote into the votes table
    :param uid: userid of the user voting
    :param pid: pid of post being voted on
    :return: None
    """
    # verify the user hasn't voted on this one
    c.execute("SELECT uid FROM votes WHERE pid = ? AND uid = ?;", (pid, uid))
    rows = c.fetchall()
    if len(rows) != 0:
        print("You have already voted for this post")
        return -1

    # generate a unique and random vno (NB:type integer)
    vno = r.randint(0, 999)
    c.execute("SELECT DISTINCT vno FROM votes WHERE pid = ?;", (pid,))
    vnos = c.fetchall()
    while vno in vnos:
        vno = (r.randint(0, 999))

    # get our current date
    currentDate = datetime.today().strftime("%Y-%m-%d")

    # add our data to the votes table
    insert_votes = '''INSERT INTO votes(pid, vno, vdate, uid) VALUES(?,?,?,?)'''
    v_data = [(pid, vno, currentDate, uid)]
    c.executemany(insert_votes, v_data)
    conn.commit()

    # print a closing statement if no errors
    print("Vote with vno {} for post {} submitted".format(vno, pid))


def filter(uid):
    """
    Filters tasks between users i.e privileged users have more tasks options
    And it calls the respective task functions when a task is selected
    :param uid: userid of the user performing the task
    :return: None
    """
    c.execute("select * from privileged;")
    results = c.fetchall()
    z = []
    for i in results:
        z.append(i[0])
    conn.commit()
    variable = True
    if uid in z:
        variable2 = True
        while variable2:
            print("""
                    \t \t \t Select an option to perform the following task -
                    \t \t \t 1 - Post a question
                    \t \t \t 2 - Search for posts
                    \t \t \t 3 - Post action-Answer
                    \t \t \t 4 - Post action-Vote
                    \t \t \t 5 - Post action-Mark as the accepted
                    \t \t \t 6 - Post action-Give a badge
                    \t \t \t 7 - Post action-Add a tag
                    \t \t \t 8 - Post Action-Edit
                    \t \t \t 9 - Logout
                    """)
            try:
                input_user = int(input())
            except:
                print("Invalid Input")
                continue

            if input_user == 1:
                post(uid)
            elif input_user == 2:
                # to be reviewed
                search()
            elif input_user == 3:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM questions")
                questions = c.fetchall()
                if (pid_selected,) in questions:
                    answer(uid, pid_selected)
                else:
                    print("Selected post is not a question")
            elif input_user == 4:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM posts")
                posts = c.fetchall()
                if (pid_selected,) in posts:
                    vote(uid, pid_selected)
            elif input_user == 5:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM answers")
                answers = c.fetchall()
                if (pid_selected,) in answers:
                    mark(uid, pid_selected)
                else:
                    print("Selected pid is not an answer")
            elif input_user == 6:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM posts")
                posts = c.fetchall()
                if (pid_selected,) in posts:
                    badge(uid, pid_selected)
            elif input_user == 7:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM posts")
                posts = c.fetchall()
                if (pid_selected,) in posts:
                    tag(uid, pid_selected)
            elif input_user == 8:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM posts")
                posts = c.fetchall()
                if (pid_selected,) in posts:
                    edit(pid_selected)
            elif input_user == 9:
                print("Goodbye")
                variable2 = False
                menu()
                break

            else:
                print("Invalid Input")
                continue
    else:
        while variable:
            print("""
                    \t \t \t Select an option to perform the following task - 
                    \t \t \t 1 - Post a question
                    \t \t \t 2 - Search for posts
                    \t \t \t 3 - Post action-Answer
                    \t \t \t 4 - Post action-Vote
                    \t \t \t 5 - Logout
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
                c.execute("SELECT DISTINCT pid FROM questions")
                questions = c.fetchall()
                if (pid_selected,) in questions:
                    answer(uid, pid_selected)
            elif input_user == 4:
                pid_selected = search()
                c.execute("SELECT DISTINCT pid FROM posts")
                posts = c.fetchall()
                if (pid_selected,) in posts:
                    vote(uid, pid_selected)
            elif input_user == 5:
                print("Goodbye")
                variable = False
                menu()
                break
            else:
                print("Invalid Input")
                continue


def user(uid):
    """
    This function checks whether the uid is present in the database
    :param uid
    :return:cond
    """
    cond = True
    find_user = "SELECT * FROM users WHERE uid = ?"
    c.execute(find_user, (uid))
    results = c.fetchall()
    if len(results) != 0:
        return cond
    else:
        return not cond


def maincheck(uid, pwd):
    """
    This function checks whether the uid and the matching password is present in the database
    :param uid: uid of user
    :param pwd: password
    :return: cond
    """
    cond = True
    find_user = "SELECT * FROM users WHERE uid = ? and pwd = ?"
    c.execute(find_user, (uid, pwd))
    results = c.fetchall()
    if len(results) != 0:
        return cond
    else:
        return not cond


def login():
    """
    Allows a user to login by checking for user credentials that he/she enters
    The user is first asked to enter the uid and then a password check is done
    The password is non-visible and if the correct password is entered a welcome message is displayed
    Else the user is asked to retry
    If the uid is not present in the database the user is asked to signup
    After a succesful login the filter function is called which displays the options for the user to choose from
    :param:  None
    :return: None
    """
    while True:
        uid = (input("Enter a uid (uid starts with u and the length is 4): "))
        findUser = "SELECT * FROM users WHERE uid =  ?"
        c.execute(findUser, (uid,))
        # if the userid is in the users table ask  the user to try something unique again
        find = "select distinct uid from users"
        c.execute(find)
        r = c.fetchall()
        z = []
        for i in r:
            z.append(list(i))
        k = []
        k.append(uid)
        while k not in z:
            uid = (input("Uid not in system. Please try again : "))
            k = []
            k.append(uid)
        while len(uid) != 4: #or (uid[0] != "u"):
            uid = (input("Enter a uid (uid starts with u and the length is 4): "))
        if user:
            password = getpass.getpass(prompt='Enter your password ')
            c.execute("SELECT * from users where uid = ? and pwd = ?", (uid, password))
            results = c.fetchall()
            if len(results) != 0:
                for i in results:
                    print("\t \t \t \t Welcome ", i[1])
                    filter(uid)
            else:
                k = True
                r = ""
                while k:
                    print("Password is incorrect ")
                    pwd = getpass.getpass(prompt='Enter your password again ')
                    if maincheck(uid, pwd):
                        k = False
                        r = pwd
                    else:
                        continue
                c.execute("SELECT * from users where uid = ? and pwd = ?", (uid, r))
                results = c.fetchall()
                if len(results) != 0:
                    for i in results:
                        print("\t \t \t \t Welcome ", i[1])
                        filter(uid)

        else:
            print("Username not recognized")
            again = input("Do you want to sign up?(y/n)")
            if again.lower() == "n":
                print("Goodbye")
                time.sleep(1)
                break
            else:
                signup()
        conn.commit()


def signup():
    """
    Allows a user to signup by adding the user's data provided into the user tables
    :param:  None
    :return: uid of the user
    """
    found = 0
    while found == 0:
        userid = (input("Enter a uid (uid starts with u and the length is 4): "))
        while len(userid) != 4: #or (userid[0] != "u"):
            userid = (input("Enter a uid (uid starts with u and the length is 4): "))
        findUser = "SELECT * FROM users WHERE uid =  ?"
        c.execute(findUser, (userid,))
        # if the userid is in the users table ask  the user to try something unique again
        find = "select distinct uid from users"
        c.execute(find)
        r = c.fetchall()
        z = []
        for i in r:
            z.append(list(i))
        k = []
        k.append(userid)
        while k in z:
            print("uid is taken : Try another one  ")
            userid = (input("Enter a uid (uid starts with u and the length is 4): "))
            k = []
            k.append(userid)

        # ask for the name and password
        name = input("Enter your name: ")
        password = input("Enter your password: ")
        password2 = input("Re-enter your password: ")

        # ensure the user has entered the same password twice else reprompt to enter password
        while password != password2:
            print("Your passwords did not match. Please try again")
            password = input("Enter your password: ")
            password2 = input("Re-enter your password: ")

        userCity = input("Which city do you live in? ")
        currentDate = datetime.today().strftime("%Y-%m-%d")

        # insert all the data unto the users table
        inserted_data = [(userid, name, password, userCity, currentDate)]
        insertData = '''INSERT INTO users(uid,name,pwd,city,crdate)
        VALUES(?,?,?,?,?)'''

        c.executemany(insertData, inserted_data)
        conn.commit()
        return userid


def menu():
    """
    Initial login screen and it calls the login,signup functions depending on the user choice
    :param:  None
    :return: None
    """
    keep = []
    while True:
        print("\t \t \t" + "Welcome!")
        my_menu = ("\t \t \t" + '''Select an option from:
        \t \t \t 1 - Sign up
        \t \t \t 2 - Login
        \t \t \t 3 - Logout    ''')
        userChoice = (input(my_menu))
        if userChoice not in keep:
            keep.append(userChoice)  # keep track of the user choices
            if str(userChoice) == "1":
                signup()
                print("\t \t \t Thank you for signing up")
            elif str(userChoice) == "2":
                login()
            elif str(userChoice) == "3":
                print("Logged out")
                user = str(input("Do you want to exit y/n "))
                user = user.lower()
                if user[0] == "y":
                    print("Goodbye")
                    sys.exit()
                else:
                    continue
            else:
                print("Invalid input")
        else:
            print("Choose the next option which you haven't chosen before")
            continue


menu()
conn.commit()
conn.close()
