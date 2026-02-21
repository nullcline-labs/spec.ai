FROM rust:1.88 AS deps
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/server/Cargo.toml crates/server/Cargo.toml
RUN mkdir -p crates/core/src crates/server/src && \
    echo "fn main() {}" > crates/server/src/main.rs && \
    touch crates/core/src/lib.rs crates/server/src/lib.rs
RUN cargo build --release -p specai-server 2>/dev/null || true
RUN rm -rf crates/

FROM deps AS builder
COPY crates/ crates/
RUN touch crates/core/src/lib.rs crates/server/src/lib.rs crates/server/src/main.rs
RUN cargo build --release -p specai-server

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1000 specai && \
    useradd --uid 1000 --gid specai --shell /usr/sbin/nologin --create-home specai
COPY --from=builder /app/target/release/spec-ai /usr/local/bin/
USER specai
EXPOSE 3040
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3040/health || exit 1
ENTRYPOINT ["spec-ai"]
