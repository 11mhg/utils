#!/bin/bash

cd ~
docker run --runtime=nvidia -it -v $(pwd):/home/ubuntu/ gasmallah:caffe bash
