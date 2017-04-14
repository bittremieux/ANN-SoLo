# base image
FROM python:3.6

# install all dependencies
COPY openblas.conf /etc/ld.so.conf.d/openblas.conf
COPY numpy-site.cfg numpy-site.cfg
COPY build.sh build.sh
ENV HOME /root
RUN bash build.sh

# create working directory and add all files
RUN mkdir -p $HOME/src/ann-solo
WORKDIR $HOME/src/ann-solo
COPY ./src $HOME/src/ann-solo

# Cythonize code
RUN python setup.py build_ext --inplace

# perform the spectral library search
ENTRYPOINT ["python", "spectral_library.py"]
