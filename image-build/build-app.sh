#!/usr/bin/env bash
. ~/.bash_profile

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 233532778289.dkr.ecr.us-west-2.amazonaws.com


docker build -t dish-dash-chat .
docker tag dish-dash-chat:latest 233532778289.dkr.ecr.us-west-2.amazonaws.com/dish-dash-chat:latest
docker push 233532778289.dkr.ecr.us-west-2.amazonaws.com/dish-dash-chat:latest