FROM python:3.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       libsdl2-dev \
       cmake \
       python3-pip \
       python3-opencv 

COPY requirements.txt /tmp/

RUN pip3 install -r /tmp/requirements.txt

RUN rm /tmp/requirements.txt