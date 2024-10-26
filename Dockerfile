FROM python:2.7.12-slim

# Set the working directory
WORKDIR /srv

# Copy code into the container
COPY . /srv
RUN python get-pip.py
# Install TensorFlow 1.3.0 and other dependencies
RUN pip install --no-cache-dir tensorflow==1.3.0 keras==2.0.8 Pillow==5.0.0 h5py==2.7.0 opencv-python==3.4.2.17
RUN pip install jupyter[notebook] ipywidgets==7.5
RUN pip install matplotlib tensorflow-datasets keras_preprocessing streamlit keras_applications
EXPOSE 8888
# jupyter notebook  --allow-root --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
CMD ["jupyter", "notebook","--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888","--NotebookApp.token=''","--NotebookApp.password=''"]

# 실행명령어
# docker build -t deep .
# docker run -it -p 8888:8888 -v [deepxplore가 있는 폴더]:/srv deep
# ex) docker run -it -p 8888:8888 -v /Users/jtlim/Repos/deepxplore:/srv deep