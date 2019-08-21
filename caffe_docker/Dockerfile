FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y vim sudo

#RUN adduser --disabled-password --gecos '' ubuntu
#RUN adduser ubuntu sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER ubuntu
#WORKDIR /home/ubuntu/
#RUN chmod a+rwx /home/ubuntu/

RUN mkdir /opt/anaconda/
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b -fp /opt/anaconda/
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

RUN ls /opt/anaconda/

ENV PATH /opt/anaconda/bin:$PATH

RUN conda update -y conda
#RUN conda update -y anaconda
#RUN conda update -y --all

RUN conda install pytorch-nightly cuda80 -c pytorch
RUN pip install google google-cloud protobuf future 

RUN mkdir /cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi/
WORKDIR "/cocoapi/PythonAPI"
RUN make install

RUN mkdir /detectron
RUN git clone https://github.com/facebookresearch/detectron /detectron/
WORKDIR "/detectron/"
RUN pip install -r requirements.txt
RUN make
WORKDIR "/"


ENV PYTHONPATH /home/utils/:$PYTHONPATH

WORKDIR "/home/"
