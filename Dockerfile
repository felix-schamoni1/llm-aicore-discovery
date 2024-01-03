# FROM --platform=amd64 nvcr.io/nvidia/pytorch:23.09-py3 as base
FROM --platform=amd64 huggingface/transformers-pytorch-gpu:4.29.2 as base


# Ubuntu 22.04 including Python 3.10
#NVIDIA CUDA® 12.2.1
#NVIDIA cuBLAS 12.2.5.6
#NVIDIA cuDNN 8.9.5
#NVIDIA NCCL 2.18.5
#NVIDIA RAPIDS™ 23.08
#Apex
#rdma-core 39.0
#NVIDIA HPC-X 2.16
#OpenMPI 4.1.4+
#GDRCopy 2.3
#TensorBoard 2.9.0
#Nsight Compute 2023.2.1.3
#Nsight Systems 2023.3.1.92
#NVIDIA TensorRT™ 8.6.1.6
#Torch-TensorRT 2.0.0.dev0
#NVIDIA DALI® 1.29.0
#MAGMA 2.6.2
#JupyterLab 2.3.2 including Jupyter-TensorBoard
#TransformerEngine 0.12
#PyTorch quantization wheel 2.1.2

WORKDIR /workspace
ENV PATH="/usr/local/lib/python3.10/bin:$PATH"
RUN python3 -c "import torch; print(torch.__version__)" && pip show torch
COPY requirements.txt requirements.txt
COPY srv.py srv.py
COPY app/ app/

RUN pip install -r requirements.txt

RUN chgrp -R nogroup /workspace && \
    chmod -R 777 /workspace

ENV PYTORCH_KERNEL_CACHE_PATH=/workspace

ENTRYPOINT ["python3", "srv.py"]
