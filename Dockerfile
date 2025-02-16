FROM ollama/ollama

RUN apt-get update && \
    apt-get install -y wget software-properties-common gnupg && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/llvm.asc && \
    add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update

RUN apt-get install -y \
    clang-20 lldb-20 lld-20 clangd-20 \
    gcc-13 g++-13 \
    python3 python3.10-venv python3-pip \
    build-essential cmake && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

RUN clang-20 --version && gcc-13 --version
ENV CC=/usr/local/bin/gcc \
    CXX=/usr/local/bin/g++

COPY . /app
COPY requirements.txt /app/WhiteFox/requirements.txt
WORKDIR /app/WhiteFox
RUN python3 -m venv whitefox-env
RUN . whitefox-env/bin/activate
RUN pip install -r requirements.txt

ENV PATH="/app/WhiteFox/whitefox-env/bin:$PATH"

