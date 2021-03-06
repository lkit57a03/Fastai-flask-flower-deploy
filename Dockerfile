FROM python:2.7.16-slim-stretch

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install Werkzeug Flask numpy fastai gevent pillow h5py

EXPOSE 5000
CMD [ "python" , "app.py"]

