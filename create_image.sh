#!/usr/bin/env bash

docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t nongaussian -f Dockerfile .
docker run -it --rm -v "$(pwd)":/workspace nongaussian
