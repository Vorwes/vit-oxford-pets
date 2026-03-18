# Inference API
This directory contains the code for web server used to deploy the vision transformer.

This inference engine is built entirely in Rust to be lightweight and faster than Python with Flask/FastAPI.

## Tech Stack
- **Web Framework:** Axum
- **Inference Engine:** Candle

## System Architecture
- **The Network Layer:** Handles requests and extracts the image, while safely handling any errors without panicking.
- **Preprocessing:** Resizes the image to `224x224`, normalizes it using the model's standard deviation and mean, and converts it to a Tensor.
- **Prediction:** A forward pass through the ViT which is wrapped in an `Arc`, ensuring thread safe concurrent inference.

## API

### `POST /predict`
Runs inference on an uploaded image and returns the predicted breed.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Payload:** A single image file attached to the form field named `image`.

**Example:**
```bash
curl -X POST http://0.0.0.0:3000/predict -F "image=@dog.jpg"
```
**Response (200 OK):**
```json
{"label": "breed_name"}
```

**Error Handling:**
- `400 Bad Request`: Missing image field or invalid file type.
- `500 Internal Server Error`: Tensor shape mismatch or math panic during the forward pass.

## Running the Server:
To start the server, make sure you have the model and it's files in the `../models/vit-pets-final` directory.

Run it in release mode for maximum speed:
```bash
cargo run --release
```
