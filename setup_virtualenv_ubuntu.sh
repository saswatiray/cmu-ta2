#!/bin/bash

# If you need to install python 3.6, you can do it with this:
#sudo add-apt-repository ppa:deadsnakes/ppa
#sudo apt-get update
#sudo apt-get install python3.6 python3.6-dev

virtualenv env --python=python3.6
source env/bin/activate
pip install --upgrade pip
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git
pip install docker grpcio-tools grpcio celery sphinx
pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
