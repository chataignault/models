FROM Ubuntu:22.04
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env
RUN git clone https://github.com/chataignault/models
WORKDIR ./models/pdevnet
RUN source install_script.sh # C++ build dependencies
RUN uv sync
RUN uv pip install signatory==1.2.6.1.9.0
COPY devnet.wheel .
RUN uv pip install --system ./devnet.wheel
CMD ["uv", "run", "grid_search"]

