# Stage 1: Builder
FROM python:3.11-slim as builder

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set up the build environment
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Install dependencies, including dev dependencies to build the wheel
RUN poetry install --no-root

# Copy the application source code
COPY src ./src

# Build the wheel
RUN poetry build -f wheel -n


# Stage 2: Production
FROM python:3.11-slim as production

# Create a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# Install only runtime dependencies
COPY --from=builder /app/poetry.lock /app/pyproject.toml /home/appuser/
RUN pip install --no-cache-dir poetry==1.8.3 && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl .

# Install the application wheel
RUN pip install --no-cache-dir *.whl

# Set the entrypoint
ENTRYPOINT ["cytoflow-qc"]
CMD ["--help"]
