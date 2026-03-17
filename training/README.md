# Model Training
This directory contains the python environment, which has the model training and data preprocessing.

## Dataset
The model was fine tuned on the **Oxford IIIT Pet** dataset, which contains images of 37 classes of cat and dog breeds (25 dog breeds, 12 cat breeds) with ~200 images each.

- **Train/Test Split:** The dataset's original 50/50 split was perserved.
- **Image Augmentations:** Random Horizontal Flip, Random Resized Crop.
- **Normalization:** 0.5 std, 0.5 mean (the same as the base model).

## Model Architecture
Google's `vit-base-patch16-224-in21k` was used as the base model.

## Hyperparameters
- **Epochs:** 3
- **Warmup Steps:** 0.1
- **Learning Rate:** 5e-5

## Evaluation
- **Test Accuracy:** 89.83%

## Exporting the Model
The model is saved as a `model.safetensors` file along with `config.json` and `preprocessor_config.json`, which are then loaded in Rust.
