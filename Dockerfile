FROM python:3.8.6
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt