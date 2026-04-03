FROM rust:1.93-bookworm AS builder
WORKDIR /app

COPY serving/Cargo.toml .
COPY serving/Cargo.lock .

RUN mkdir src
RUN echo "fn main() {}" > src/main.rs
RUN cargo build --release

COPY serving/src/ src/
RUN touch src/main.rs
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app

COPY models/vit-pets-final/ /app/models/vit-pets-final
COPY --from=builder /app/target/release/serving .

ENV MODEL_PATH=./models/vit-pets-final

EXPOSE 3000
CMD ["./serving"]
