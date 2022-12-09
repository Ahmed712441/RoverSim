FROM python:3.7-slim-buster
WORKDIR /app
COPY  ./requirments.txt .
RUN apt-get update
RUN apt-get install -y libopencv-dev python3-opencv
RUN [ "pip3","install","-r","requirments.txt" ]
RUN mkdir code
COPY ./code ./code
RUN pip install markupsafe==2.0.1
COPY ./code/mask2.jpg .
COPY ./calibration_images/map_bw.png .
