FROM nvidia/cuda:11.4.0-base-ubuntu20.04

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3.9 python3-pip

RUN ln -s /usr/bin/python3.9 /usr/bin/python

# upgrade pip
RUN python -m pip -q install pip --upgrade

COPY ./requirements.txt /home/root/thesis/requirements.txt
WORKDIR /home/root/thesis
RUN ls
RUN python -m pip install -r requirements.txt

COPY . /home/root/thesis
RUN python setup.py install

ENV WANDB_LOG_MODEL=true

CMD python -m src.main --debug