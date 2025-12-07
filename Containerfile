FROM docker.io/python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install grpcurl
RUN curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz \
    | tar -xz --no-same-owner -C /usr/local/bin

# Install Python dependencies
RUN pip install --no-cache-dir \
    grpcio>=1.60.0 \
    grpcio-tools>=1.60.0 \
    protobuf>=4.25.0 \
    pandas>=2.1.0 \
    pyarrow>=14.0.0 \
    duckdb>=1.0.0 \
    tqdm>=4.66.0

# Copy proto file and compile
COPY proto/lute.proto /app/proto/
RUN python -m grpc_tools.protoc \
    -I/app/proto \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/proto/lute.proto

# Copy scripts
COPY scripts/extract_from_lute.py /app/
COPY scripts/duckdb_query.py /app/

# Default to extraction
ENTRYPOINT ["python", "extract_from_lute.py"]
