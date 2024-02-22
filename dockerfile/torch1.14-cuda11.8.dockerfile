FROM nvcr.io/nvidia/pytorch:22.12-py3

# Ubuntu: 20.04
# Python: 3.8
# CUDA: 11.8.0
# cuDNN: 8.7.0.84
# NCCL: v2.16.2-1 + FP8 Support
# PyTorch: 1.14.0a0+410ce96

LABEL maintainer="MS-AMP"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    bc \
    build-essential \
    curl \
    dmidecode \
    git \
    iproute2 \
    jq \
    moreutils \
    net-tools \
    openssh-client \
    openssh-server \
    sudo \
    util-linux \
    vim \
    wget \
    python3-mpi4py \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ARG NUM_MAKE_JOBS=
ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /opt/msamp

ADD third_party third_party
RUN cd third_party/msccl && \
    make -j ${NUM_MAKE_JOBS} src.build NVCC_GENCODE="\
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_90,code=sm_90" && \
    make install
# cache TE build to save time in CI
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.1

ADD . .
RUN python3 -m pip install . && \
    make postinstall

ENV LD_PRELOAD="/usr/local/lib/libmsamp_dist.so:/usr/local/lib/libnccl.so:${LD_PRELOAD}"

# Set up entrypoint
COPY dockerfile/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
