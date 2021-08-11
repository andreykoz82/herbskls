FROM pytorch/pytorch
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt