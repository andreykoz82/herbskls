FROM pytorch/pytorch
ADD . /python-flask
WORKDIR /python-flask
EXPOSE 5000
ENV FLASK_APP=app.py
RUN pip install -r requirements.txt
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]