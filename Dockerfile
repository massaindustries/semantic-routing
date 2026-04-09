# MyModel Dockerfile — Single HTTP proxy service (no Envoy)
# Multi-stage build: Rust ML libraries → Go binary → Runtime

# ─── Stage 1: Build Rust candle-binding (ML embeddings) ───
ARG BUILDPLATFORM
FROM --platform=$BUILDPLATFORM rust:1.90-bookworm AS rust-builder

RUN apt-get update && apt-get install -y \
    make build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
ENV CARGO_INCREMENTAL=1
ENV CARGO_PROFILE_RELEASE_LTO=thin

WORKDIR /app

# Cache Rust dependencies
COPY candle-binding/Cargo.toml candle-binding/Cargo.loc[k] ./candle-binding/
RUN cd candle-binding && \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    cargo build --release --no-default-features && \
    rm -rf src

# Build candle-binding
COPY candle-binding/src/ ./candle-binding/src/
RUN cd candle-binding && \
    find src -name "*.rs" -exec touch {} + && \
    cargo build --release --no-default-features

# ─── Stage 1b: Build ml-binding (Linfa ML) ───
FROM --platform=$BUILDPLATFORM rust:1.90-bookworm AS ml-builder

RUN apt-get update && apt-get install -y build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

WORKDIR /app

COPY ml-binding/Cargo.toml ml-binding/Cargo.loc[k] ./ml-binding/
RUN cd ml-binding && \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    cargo build --release && rm -rf src

COPY ml-binding/src/ ./ml-binding/src/
RUN cd ml-binding && \
    find src -name "*.rs" -exec touch {} + && \
    cargo build --release

# ─── Stage 1c: Build nlp-binding (BM25 + N-gram) ───
FROM --platform=$BUILDPLATFORM rust:1.90-bookworm AS nlp-builder

RUN apt-get update && apt-get install -y build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

WORKDIR /app

COPY nlp-binding/Cargo.toml nlp-binding/Cargo.loc[k] ./nlp-binding/
RUN cd nlp-binding && \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    cargo build --release && rm -rf src

COPY nlp-binding/src/ ./nlp-binding/src/
RUN cd nlp-binding && \
    find src -name "*.rs" -exec touch {} + && \
    cargo build --release

# ─── Stage 2: Build Go binary ───
FROM --platform=$BUILDPLATFORM golang:1.24-bookworm AS go-builder

# Install dependencies for CGO and linking
RUN apt-get update && apt-get install -y libssl-dev build-essential pkg-config file && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a centralized folder for all .so libraries
RUN mkdir -p /app/libs

# Copy module files and binding sources (including .h headers)
COPY candle-binding/ ./candle-binding/
COPY ml-binding/ ./ml-binding/
COPY nlp-binding/ ./nlp-binding/
COPY src/semantic-router/go.mod src/semantic-router/go.sum ./src/semantic-router/

# Copy compiled libraries from Rust stages to centralized libs folder
COPY --from=rust-builder /app/candle-binding/target/release/libcandle_semantic_router.so /app/libs/
COPY --from=ml-builder /app/ml-binding/target/release/libml_semantic_router.so /app/libs/
COPY --from=nlp-builder /app/nlp-binding/target/release/libnlp_binding.so /app/libs/

# Download Go dependencies
RUN cd src/semantic-router && go mod download

# Copy Go source code
COPY src/semantic-router/ ./src/semantic-router/

ENV CGO_ENABLED=1
ENV GOOS=linux
# Set library path for the linker during build time
ENV LD_LIBRARY_PATH=/app/libs

# Build the proxy binary
# CGO_CFLAGS includes binding directories for header discovery
RUN cd src/semantic-router && \
    CGO_CFLAGS="-I/app/candle-binding -I/app/ml-binding -I/app/nlp-binding" \
    CGO_LDFLAGS="-L/app/libs -lcandle_semantic_router -lml_semantic_router -lnlp_binding" \
    go build -ldflags="-w -s" -o /app/bin/mymodel cmd/main.go

# ─── Stage 3: Runtime ───
FROM debian:bookworm-slim

# Install runtime dependencies and Python for HuggingFace
RUN apt-get update && apt-get install -y \
    ca-certificates \
    openssl \
    python3 \
    python3-pip \
    && pip3 install --break-system-packages --no-cache-dir huggingface_hub[cli] \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=go-builder /app/bin/mymodel /app/mymodel

# Copy .so libraries to system path and update linker
COPY --from=go-builder /app/libs/*.so /usr/local/lib/
RUN ldconfig

# Copy configuration
COPY config/config.yaml /app/config/

# Expose required ports
# MyModel HTTP proxy port
EXPOSE 8000
# Metrics port
EXPOSE 9190
# API server port
EXPOSE 8080

ENTRYPOINT ["/app/mymodel", "--config", "/app/config/config.yaml"]
