FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="@hirune924"

RUN apt-get update --fix-missing && apt-get install -y \
    git libopencv-dev python-openslide unzip 
    #&& \
    #apt-get clean && \
    #apt-get autoremove && \
    #rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex

#RUN pip uninstall -y pillow && \
RUN pip install --upgrade pip && \
    pip install pandas scikit-image opencv-python openslide-python \
                scikit-learn pytorch-lightning imagecodecs tifffile \
                pretrainedmodels catalyst[all] segmentation-models-pytorch timm hydra-core \
                scikit-plot && \
    pip install -U neptune-client && \
    pip install -U git+https://github.com/albumentations-team/albumentations

