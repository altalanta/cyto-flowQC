FROM python:3.11-slim

ENV POETRY_VERSION=1.8.3 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CYTOFLOW_QC_HOME=/opt/cytoflow-qc \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR ${CYTOFLOW_QC_HOME}

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-root

COPY src src
COPY configs configs
COPY samplesheets samplesheets
COPY notebooks notebooks

RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "cytoflow-qc"]
CMD ["--help"]
