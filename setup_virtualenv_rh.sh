#!/bin/bash

# Not necessarily functional since getting the right version of python3.6 on CentOS is a PITA
# and there appear to be multiple options.  x_X

yum install python36-devel
virtualenv-3 env --python=python36
source env/bin/activate
pip install --upgrade pip
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git
pip install docker grpcio-tools grpcio celery sphinx

pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
