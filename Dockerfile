# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set up the build environment
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Install dependencies (only main dependencies for building)
RUN poetry install --only main --no-root

# Copy the application source code
COPY src ./src

# Build the wheel
RUN poetry build -f wheel -n


# Stage 2: Production
FROM python:3.11-slim AS production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl .

# Switch to non-root user for installing packages
USER appuser

# Install the application wheel in user space
RUN pip install --user --no-cache-dir *.whl && \
    rm -f *.whl

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:$PATH"

# Set the entrypoint
ENTRYPOINT ["cytoflow-qc"]
CMD ["--help"]
