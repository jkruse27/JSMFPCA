FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git python3-venv \
    python3-dev \
    build-essential

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt -y install cm-super

RUN apt-get install -y poppler-utils

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get update && \
    apt -y install texlive texlive-latex-extra texlive-fonts-recommended dvipng

RUN apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y ttf-mscorefonts-installer && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

ENV VIRTUAL_ENV=/home/user/venv
RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]