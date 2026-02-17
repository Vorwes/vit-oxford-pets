import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop
from transformers import (
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


def transform_train(batch):
    augmented_images = [
        augment_transforms(img.convert("RGB")) for img in batch["image"]
    ]

    inputs = processor(
        augmented_images,
        return_tensors="pt",
        do_resize=False,
    )

    inputs["labels"] = batch["label"]

    return inputs


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
    dataset = load_dataset("timm/oxford-iiit-pet", cache_dir="../data/hf_cache")

    test_data = dataset["test"]
    train_val_data = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_val_data["train"]
    val_data = train_val_data["test"]

    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)

    target_size = processor.size["height"]

    augment_transforms = Compose(
        [
            RandomResizedCrop(target_size),
            RandomHorizontalFlip(),
        ]
    )

    train_data.set_transform(transform_train)
    val_data.set_transform(transform_val)
    test_data.set_transform(transform_val)

    labels = dataset["train"].features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        output_dir="../models/vit-pets",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_steps=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    output_dir = "../models/vit-pets-final"

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Model saved.")
