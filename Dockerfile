FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN python3 -m pip install --upgrade pip cython
RUN python3 -m pip install torch==1.5.0 torchvision==0.6.0 mxnet==1.6.0 gluoncv==0.7.0 Keras==2.3.1 segmentation-models==1.0.1 \
 progressbar2==3.38.0 tensorflow-probability==0.11.0 dm-sonnet==2.0.0 opencv-python==4.2.0.34 Shapely==1.7.0 moviepy==1.0.2 \
 kornia==0.3.1 lap==0.4.0 Cython==0.29.15 cython-bbox==0.1.3 pandas==0.25.3 albumentations>=0.3.0

RUN apt-get update && apt-get install -yq libsm6 libxext6 libxrender-dev
RUN python3 -m pip uninstall -y mxnet
RUN python3 -m pip install mxnet-cu101==1.7.0
WORKDIR /app