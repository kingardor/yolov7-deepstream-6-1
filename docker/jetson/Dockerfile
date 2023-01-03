# Use L4T tensorrt docker listed on https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt/tags
# Use r8.4.1 for DS 6.1.1
FROM nvcr.io/nvidia/l4t-tensorrt:r8.4.1-runtime

#Install vpi-dev and vpi-lib
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libnvvpi2 vpi2-dev vpi2-samples && \
        rm -rf /var/lib/apt/lists/* && \
        apt autoremove

# Install dependencies
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        rsyslog git \
        tzdata \
        libgstrtspserver-1.0-0 \
        libjansson4 \
        libglib2.0-0 \
        libjson-glib-1.0-0 \
        librabbitmq4 \
        gstreamer1.0-rtsp \
        libyaml-cpp-dev \
        libyaml-cpp0.6 \
        libcurl4-openssl-dev \
        ca-certificates && \
        rm -rf /var/lib/apt/lists/* && \
        apt autoremove


# adding Cuda missing symlinks in base docker
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcufft.so.10 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcufft.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcublas.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcublas.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libcudnn.so.8 /usr/lib/aarch64-linux-gnu/libcudnn.so
RUN ln -s /usr/local/cuda-11.4/lib64/libcudart.so.11.0 /usr/local/cuda-11.4/lib64/libcudart.so
# Nvinfer libs:
RUN ln -s /usr/lib/aarch64-linux-gnu/libnvinfer.so.8 /usr/lib/aarch64-linux-gnu/libnvinfer.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libnvparsers.so.8 /usr/lib/aarch64-linux-gnu/libnvparsers.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.8 /usr/lib/aarch64-linux-gnu/libnvonnxparser.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so.8 /usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so
# NPP libs:
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppc.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppc.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppial.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppial.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppicc.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppicc.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppidei.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppidei.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppif.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppif.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppig.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppig.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppim.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppim.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppist.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppist.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppisu.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppisu.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppitc.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnppitc.so
RUN ln -s /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnpps.so.11 /usr/local/cuda-11.4/targets/aarch64-linux/lib/libnpps.so


RUN ldconfig

# Install DeepStreamSDK using tar package.
ENV DS_REL_PKG deepstream_sdk_v6.1.0_jetson.tbz2

COPY "${DS_REL_PKG}"  \
/

RUN DS_REL_PKG_DIR="${DS_REL_PKG%.tbz2}" && \
cd / && \
tar -xvf "${DS_REL_PKG}" -C / && \
cd /opt/nvidia/deepstream/deepstream && \
./install.sh && \
cd / && \
rm -rf "/${DS_REL_PKG}"

RUN ldconfig

CMD ["/bin/bash"]
WORKDIR /opt/nvidia/deepstream/deepstream

ENV LD_LIBRARY_PATH /usr/local/cuda-11.4/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
