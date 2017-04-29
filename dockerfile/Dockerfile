FROM gcr.io/tensorflow/tensorflow
MAINTAINER nekonyanneko

RUN apt-get update
RUN apt-get -y install vim
RUN apt-get -y install build-essential
RUN apt-get -y install git
RUN pip install keras
RUN git clone https://github.com/facebookresearch/fastText.git
RUN cd fastText;make
RUN easy_install gensim
