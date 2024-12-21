FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env
RUN git clone https://github.com/chataignault/models
WORKDIR /models/pdevnet
RUN uv sync
RUN uv pip install signatory==1.2.6.1.9.0
RUN uv pip install --system ./devnet.wheel
CMD ["python", "-m", "grid_search"]

