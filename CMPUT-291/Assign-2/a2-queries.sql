.echo on 

--Question 1
select distinct users.uid from users,ubadges,posts,badges
where badges.bname = ubadges.bname and badges.type = 'gold' and ubadges.uid = users.uid and title like "%?"
and posts.poster = users.uid;

--Question 2
select distinct posts.pid, title from posts,tags
where lower(tags.tag) = "relational" and lower(title) like "%?%" and tags.pid = posts.pid
INTERSECT
select distinct posts.pid, title from posts,tags
where lower(tags.tag) = "database" and lower(title) like "%?%" and tags.pid = posts.pid
UNION
select distinct posts.pid, title from posts,tags
where lower(title) like "%relational database%?%" and tags.pid = posts.pid;

--Question 3
SELECT DISTINCT posts.pid,posts.title from posts,questions,answers
where questions.pid = posts.pid
EXCEPT
select DISTINCT p1.pid,p1.title from posts p1,posts p2,questions,answers
where p2.pid = answers.pid and p1.pid = questions.pid and questions.pid = answers.qid
AND julianday(p2.pdate) - julianday(p1.pdate) <= 3;


--Question 4
select DISTINCT p1.poster from posts p1,posts p2,questions,answers
where questions.pid = p1.pid and answers.qid = questions.pid
group by p1.poster
having COUNT(DISTINCT questions.pid) > 2 and COUNT(DISTINCT answers.pid) > 2;

--Question 5
select p1.poster from posts p1,posts p2,questions,answers,votes v1,votes v2
where questions.pid = p1.pid and answers.pid = p2.pid and p2.poster = p1.poster and v1.pid = p1.pid and v2.pid = p2.pid
group by p1.poster
having SUM(DISTINCT v1.vno) + SUM(DISTINCT V2.vno) > 4;


--Question 6
SELect tags.tag,COUNT(votes.vno),count(DISTINCT posts.pid) from tags,posts,votes
where votes.pid = posts.pid AND tags.pid = posts.pid
group by tags.tag
ORDER BY COUNT(votes.vno) DESC LIMIT 3;

--Question 7
select p1.pdate,t1.tag,COUNT(t1.tag) as val from tags t1, tags t2,posts p1
where p1.pid = t1.pid and t1.tag = t2.tag
GROUP by p1.pdate,t1.tag;

--Question 8
select users.uid, COUNT(DISTINCT p1.pid),COUNT(DISTINCT p2.pid),count(DISTINCT v1.vno),COUNT(distinct v2.vno)
from users, questions,answers
left join posts p2 on p2.poster = users.uid and answers.qid = p2.pid or questions.theaid = answers.pid
LEFT JOIN posts p1 on P1.poster = users.uid and questions.pid = p1.pid
left join votes as v1 on v1.uid = users.uid
LEFT Join votes as v2 on v2.uid = users.uid or v2.pid = p2.pid
GROUP BY users.uid;

--Question 9
/*create view questionInfo(pid, uid, theaid, voteCnt,ansCnt) as select DISTINCT q1.pid,p1.poster,q1.theaid,COUNT(DISTINCT votes.vno),COUNT(DISTINCT answers.qid)
from posts p1,questions q1
LEFT join answers on q1.pid = answers.qid
LEFT JOIN votes on votes.pid = q1.pid
where q1.pid = p1.pid AND p1.pdate >  datetime('2020-09-01')
GROUP BY q1.pid;*/

--Question 10
select users.city,count(users.city) from users
group by users.city;
