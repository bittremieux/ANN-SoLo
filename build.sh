#!/bin/bash

####################################################################################################

# install OpenBLAS (from: https://github.com/ogrisel/docker-openblas)
mkdir $HOME/build
cd $HOME/build

apt-get clean && apt-get -y update
apt-get -y install cmake
apt-get -y install git-core build-essential gfortran libboost-all-dev libgsl0-dev libeigen3-dev

# get the stable releases from the master branch
git clone -q --branch=master git://github.com/xianyi/OpenBLAS.git
(cd OpenBLAS && git checkout tags/v0.2.18 && make DYNAMIC_ARCH=1 NO_AFFINITY=1 NUM_THREADS=32 && make install)

# rebuild ld cache, this assumes that: /etc/ld.so.conf.d/openblas.conf was installed by Dockerfile and that the libraries are in /opt/OpenBLAS/lib
ldconfig

####################################################################################################

# install NumPy with OpenBLAS support (from: https://github.com/ogrisel/docker-sklearn-openblas)
cd $HOME/build

# install Cython
pip install Cython==0.24.1

git clone -q --branch=master git://github.com/numpy/numpy.git
cp /numpy-site.cfg numpy/site.cfg
(cd numpy && git checkout tags/v1.11.1 && python setup.py install)

####################################################################################################

# install Python packages through pip
cd $HOME/build

# Python 2/3 compatibility
pip install functools32==3.2.3-2
pip install six==1.10.0
# Pandas
pip install Bottleneck==1.1.0
pip install pandas==0.18.1
# NumExpr
pip install numexpr==2.6.1
# Annoy
pip install git+git://github.com/spotify/annoy.git
# Scikit-Learn
pip install scikit-learn==0.17.1
# I/O
pip install ConfigArgParse==0.10.0
pip install lxml==3.6.1
pip install pathlib2==2.1.0
pip install pyteomics==3.3.1
pip install tqdm==4.8.3
pip install joblib==0.10.0

####################################################################################################

# install nmslib
cd $HOME/build

git clone -q --branch=master git://github.com/searchivarius/nmslib.git
(cd nmslib/similarity_search && git checkout tags/v1.5.3 && cmake . && make && cd ../python_vect_bindings && make install)

####################################################################################################

# minimize image size
apt-get remove -y --purge git-core build-essential
apt-get autoremove -y
apt-get clean -y

# clean-up
cd /
rm -rf /build.sh
rm -rf /numpy-site.cfg
