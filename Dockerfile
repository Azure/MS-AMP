FROM nvcr.io/nvidia/pytorch:22.09-py3

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Install FP8-NCCL (arch >= 5.3)
RUN cd /tmp && git clone -b ziyyang/fp8-support https://github.com/yzygitzh/nccl && \
    cd nccl && \
    make -j src.build NVCC_GENCODE="\
    -gencode=arch=compute_53,code=sm_53 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_62,code=sm_62 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_72,code=sm_72 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_87,code=sm_87 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90,code=sm_90 \
    " && \
    make install && \
    rm -rf /tmp/nccl
