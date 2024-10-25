FROM ubuntu:latest
LABEL authors="jtlim"

ENTRYPOINT ["top", "-b"]