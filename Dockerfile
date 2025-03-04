FROM ollama/ollama

USER root

RUN apt-get update && \
    apt-get install -y wget software-properties-common gnupg curl apt-transport-https && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/llvm.asc && \
    echo "deb [arch=arm64] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list.d/llvm-17.list && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update

RUN apt-get install -y \
    clang-17 lldb-17 lld-17 clangd-17 \
    gcc-12 g++-12 \
    build-essential cmake git && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100

RUN apt-get install -y \
    git tmux python3 python3-pip python3-venv \
    curl wget vim

RUN clang --version && gcc --version

# WORKDIR /app
# RUN git clone https://github.com/llvm/llvm-project.git && \
#     cd llvm-project && \
#     mkdir build && \
#     cd build && \
#     cmake -G "Unix Makefiles" \
#           -DLLVM_ENABLE_PROJECTS="clang" \
#           -DCMAKE_BUILD_TYPE=Release \
#           -DLLVM_USE_LINKER=lld \
#           -DCMAKE_C_COMPILER=clang \
#           -DCMAKE_CXX_COMPILER=clang++ \
#           ../llvm && \
#     make -j$(nproc)

ENV CC=clang \
    CXX=clang++

COPY WhiteFox /app/WhiteFox
WORKDIR /app/WhiteFox
RUN pip3 install -r requirements.txt