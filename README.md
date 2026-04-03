---
title: ViT Oxford Pets
emoji: 🐕
sdk: docker
---
# **Vision Transformer (ViT) Pet Classifier**

This is an end-to-end Machine Learning project where I fine-tuned a vision transformer (google's vit-base-patch16-224-in21k) on the Oxford-IIIT Pet dataset using Python, and then served this fine-tuned model via a highly optimized asynchronous Rust backend.

---
## **Project Structure**
- `/training` holds the training and testing code (**Python**).
- `/serving` holds the server code (**Rust**).

----
## **Usage**
1. Clone the repository.
2. Setup the Python environment (requires [uv](https://docs.astral.sh/uv/)):
```bash
cd training
uv sync
```
3. Run the training script:
```bash
uv run python -m src.train
```
4. Run the server (running it with --release is necessary as it significantly speeds up the server):
```bash
cd serving
cargo run --release
```
5. Now the API is accessible at `http://0.0.0.0:3000/predict`.
6. You can then test it using:
```bash
curl -X POST http://localhost:3000/predict -F "image=@test_image.jpg"
```
Expected Result:
```json
{"label": "breed_name"}
```
---
## **Huggingface**
The model is also available on [Huggingface](https://huggingface.co/Vorwes/vit-pet-classifier).

----
## **Specific Details**
For more details on the project, check out the README.md files in the `/training` and `/serving` directories.
