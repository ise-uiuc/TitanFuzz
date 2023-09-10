# Build from tensorflow
FROM tensorflow/tensorflow:2.11.0rc1-gpu

COPY unparser.patch /home/

# Install pytorch and transformers
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install transformers

# Fix bug in astunparse in Python 3.8 https://github.com/simonpercivall/astunparse/pull/44/files
RUN apt update
RUN apt install -y git
RUN pip install astunparse==1.6.3
WORKDIR /usr/local/lib/python3.8/dist-packages/astunparse
RUN git apply --whitespace=fix /home/unparser.patch

WORKDIR /home/
