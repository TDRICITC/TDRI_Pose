FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
#FROM ubuntu:20.04
LABEL INSTANCE_TYPE="hello-world"
MAINTAINER gwjinag1026@gmail.com

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y locales && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING utf-8
ENV PYTHONUNBUFFERED=1

ENV TZ="Asia/Taipei"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get -y install tzdata curl vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
	
RUN apt-get update && \
    pip install opencv-python && \
    apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install mediapipe
RUN pip install cvzone
#RUN pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install pyyaml
RUN pip install tqdm
RUN pip install seaborn

## Change TimeZone
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata



EXPOSE 5000

# set work directory
WORKDIR /opt



# install dependencies
COPY ./flask_app /opt/flask_app
COPY run.py requirements.txt logging.ini ./
RUN pip install -r requirements.txt
RUN mkdir /opt/logs

ENV FLASK_APP="/opt/run.py"
ENV FLASK_ENV="development"

# CMD flask run --no-reload

