FROM ubuntu:16.04
MAINTAINER <peter@goldsborough.me>

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common apt-utils

RUN apt-get update && apt-get install -y \
    clang-3.8 git python3-numpy python3-dev python3-pip python3-wheel

RUN pip3 install --upgrade pip && pip3 install tensorflow-gpu

# Additional packages to do work.
RUN apt-get install -y vim emacs

ENV C clang-3.8
ENV CXX clang++-3.8

WORKDIR /root
