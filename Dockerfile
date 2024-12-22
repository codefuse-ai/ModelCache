FROM python:3.9-slim-bookworm

WORKDIR /home/user

COPY ./requirements.txt /home/user/docker_requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /home/user/docker_requirements.txt --retries 5 --timeout 120
