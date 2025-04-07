FROM python:3.10 as base
RUN apt-get update && \
    apt-get -y install \
    curl \
    gdal-bin \
    git \
    htop \
    libgdal-dev \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-opencv \
    python3-pip \
    python3-shapely \
    wget \
    unzip && \
    apt-get clean

FROM base as dev
WORKDIR /code
ENV PYTHONPATH /code
ADD requirements.txt /code
RUN pip install pip==24.1.2
RUN pip install GDAL==3.6.2 --no-cache-dir && \
    pip install torch==2.5.0 --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

ADD . /code
