FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation 

RUN apt-get install -yq --no-install-recommends \
    build-essential \
    emacs \
    vim \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    pandoc \
    python-dev \
    spyder3 \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-xetex \
    tzdata \
    tmux \
    unzip \
    nano 

RUN apt-get install -y libopencv-dev python3-opencv

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

RUN apt-get install -y python3 python3-pip 

RUN pip3 install tensorflow-gpu==1.13.1 
RUN pip3 install numpy matplotlib jupyter scikit-learn scipy pillow jupyterhub jupyterlab tqdm 
RUN pip3 install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall 

EXPOSE 8888
EXPOSE 6006


ENV PYTHONPATH="$PYTHONPATH:/home/utils/"

WORKDIR "/home/"


CMD /bin/bash
