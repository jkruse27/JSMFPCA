FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git python3-venv \
    python3-dev \
    build-essential

RUN apt-get update && \
    apt -y install cm-super poppler-utils

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

# Set the working directory before copying files
WORKDIR /workspace

# Copy the requirements file and your local package folder
COPY --chown=user:user requirements.txt .
COPY --chown=user:user physfunc/ ./physfunc/
COPY --chown=user:user pyproject.toml .

# Install the requirements first
RUN pip3 install --no-cache-dir -r requirements.txt

# Compile and install the local jsmfpca folder in editable mode
RUN pip3 install -e .

CMD ["/bin/bash"]