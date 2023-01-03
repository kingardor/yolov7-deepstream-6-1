docker build -t deepstream:dgpu --build-arg TENSORRT_VERSION="8.4.1-1+cuda11.6" --build-arg CUDNN_VERSION="8.4.1.50-1+cuda11.6" --build-arg CUDA_VERSION="11.7.1" .
