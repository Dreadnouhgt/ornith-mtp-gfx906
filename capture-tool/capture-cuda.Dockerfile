# CUDA port of capture.Dockerfile (gfx906/ROCm -> GB10/sm_121)
FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y cmake build-essential git libgomp1 libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
COPY llama.cpp-src /build/llamacpp
WORKDIR /build/llamacpp
ENV LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib/stubs
RUN cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/targets/sbsa-linux/lib/stubs -lcuda" -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/targets/sbsa-linux/lib/stubs -lcuda" \
    && cmake --build build --config Release -j$(nproc) --target llama-hidden-capture
RUN mkdir -p /builded && cp -P build/bin/llama-hidden-capture /builded/ && cp -P build/bin/*.so* /builded/ 2>/dev/null || true

FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS final
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /builded/ /app/
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-hidden-capture"]
