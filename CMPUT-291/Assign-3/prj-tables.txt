drop table if exists answers;
drop table if exists questions;
drop table if exists votes;
drop table if exists tags;
drop table if exists posts;
drop table if exists ubadges;
drop table if exists badges;
drop table if exists privileged;
drop table if exists users;

PRAGMA foreign_keys = ON;

create table users (
  uid		char(4),
  name		text,
  pwd		text,
  city		text,
  crdate	date,
  primary key (uid)
);
create table privileged (
  uid		char(4),
  primary key (uid),
  foreign key (uid) references users
);
create table badges (
  bname		text,
  type		text,
  primary key (bname)
);
create table ubadges (
  uid		char(4),
  bdate		date,
  bname		text,
  primary key (uid,bdate),
  foreign key (uid) references users,
  foreign key (bname) references badges
);
create table posts (
  pid		char(4),
  pdate		date,
  title		text,
  body		text,
  poster	char(4),
  primary key (pid),
  foreign key (poster) references users
);
create table tags (
  pid		char(4),
  tag		text,
  primary key (pid,tag),
  foreign key (pid) references posts
);
create table votes (
  pid		char(4),
  vno		int,
  vdate		text,
  uid		char(4),
  primary key (pid,vno),
  foreign key (pid) references posts,
  foreign key (uid) references users
);
create table questions (
  pid		char(4),
  theaid	char(4),
  primary key (pid),
  foreign key (theaid) references answers
);
create table answers (
  pid		char(4),
  qid		char(4),
  primary key (pid),
  foreign key (qid) references questions
);
