ARG BASE_ROCM_IMAGE="docker.io/mixa3607/rocm-gfx906:7.2.4-complete"
ARG ROCM_ARCH="gfx906"

FROM ${BASE_ROCM_IMAGE} AS rocm_base
RUN apt-get update && \
    apt-get install -y curl libgomp1 git python3 python3-venv numactl && \
    pip3 config set global.break-system-packages true && \
    true

ARG ROCM_ARCH
ENV AMDGPU_TARGETS=${ROCM_ARCH}

FROM rocm_base AS build_llamacpp
RUN apt-get install -y build-essential cmake libssl-dev
COPY llama.cpp-b10043 /build/llamacpp
WORKDIR /build/llamacpp

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON                 \
        -DGGML_HIP_GRAPHS=ON          \
        -DGGML_HIP_RCCL=ON            \
        -DAMDGPU_TARGETS="$ROCM_ARCH" \
        -DGGML_BACKEND_DL=ON          \
        -DGGML_RPC=ON                 \
        -DGGML_CPU_ALL_VARIANTS=ON    \
        -DGGML_AVX512=ON              \
        -DGGML_AVX512_VBMI=ON         \
        -DGGML_AVX512_VNNI=ON         \
        -DGGML_AVX512_BF16=ON         \
        -DCMAKE_BUILD_TYPE=Release    \
        -DLLAMA_BUILD_TESTS=OFF       \
    && cmake --build build --config Release -j$(nproc)
RUN mkdir -p /builded && cp -r ./build/bin/* .devops/tools.sh /builded

FROM rocm_base AS final
WORKDIR /app
COPY --from=build_llamacpp /builded/ /app
ENTRYPOINT ["/app/tools.sh"]
