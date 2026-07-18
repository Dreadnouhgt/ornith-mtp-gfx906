ARG BASE_ROCM_IMAGE="docker.io/mixa3607/rocm-gfx906:7.2.4-complete"
ARG ROCM_ARCH="gfx906"

############# Base image #############
FROM ${BASE_ROCM_IMAGE} AS rocm_base
RUN apt-get update && \
    apt-get install -y curl libgomp1 git python3 python3-venv numactl build-essential cmake libssl-dev && \
    pip3 config set global.break-system-packages true && \
    true

ARG ROCM_ARCH
ENV AMDGPU_TARGETS=${ROCM_ARCH}

############# Build #############
FROM rocm_base AS build_llamacpp
COPY llama.cpp-src /build/llamacpp
WORKDIR /build/llamacpp

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON                 \
        -DAMDGPU_TARGETS="$ROCM_ARCH" \
        -DCMAKE_BUILD_TYPE=Release    \
        -DLLAMA_BUILD_TESTS=OFF       \
    && cmake --build build --config Release -j$(nproc) --target llama-hidden-capture
RUN mkdir -p /builded && cp -P build/bin/llama-hidden-capture build/bin/*.so* /builded/

############# Final #############
FROM rocm_base AS final
WORKDIR /app
COPY --from=build_llamacpp /builded/ /app/
ENTRYPOINT ["/app/llama-hidden-capture"]
