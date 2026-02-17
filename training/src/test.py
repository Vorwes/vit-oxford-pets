import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


def transform_val(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images, return_tensors="pt")

    inputs["labels"] = batch["label"]

    return inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    test_data = load_dataset(
        "timm/oxford-iiit-pet", split="test", cache_dir="../data/hf_cache"
    )

    test_data.set_transform(transform_val)

    MODEL_PATH = "../models/vit-pets-final"

    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    print(f"Test Accuracy: {metrics['eval_accuracy']:.2%}")
    print(f"Test Loss: {metrics['eval_loss']:.4f}")

    with open("results/metrics.txt", "w") as file:
        file.write(f"Test Accuracy: {metrics['eval_accuracy']:.2%}\n")
        file.write(f"Test Loss: {metrics['eval_loss']:.4f}\n")
