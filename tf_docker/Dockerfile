FROM nvcr.io/nvidia/tensorflow:19.05-py3 

RUN apt-get update && apt-get -yq upgrade

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN python3 -m pip install numpy scipy scikit-learn opencv-contrib-python \
                jupyter ipython pillow imgaug


RUN python3 -m pip install tqdm imageio imageio-ffmpeg easydict

RUN python3 -m pip install tensorflow-gpu==1.14.0 --upgrade

EXPOSE 8888
EXPOSE 6006


ENV PYTHONPATH="$PYTHONPATH:/home/utils/"

WORKDIR "/home/"

RUN rm -rf /var/lib/apt/lists/*

CMD /bin/bash
