import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple
import zipfile
import os

PRETRAINED_MODEL_NAME: str = "distilbert-base-uncased"
TRAINING_DATASET_PATH: str = "data/rotten_tomatoes_critic_reviews.csv"
MODEL_OUTPUT_PATH: str = "./tuned_model"
EPOCHS: int = 3
DATASET_ROWS_LIMIT: int = 100000
RANDOM_STATE: int = 42


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        zip_path = file_path + ".zip"
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall("./data")

    df = pd.read_csv(file_path)
    df = df[["review_content", "review_score"]]
    df = df.dropna(subset=["review_content", "review_score"])
    return df.sample(DATASET_ROWS_LIMIT, random_state=RANDOM_STATE)


def preprocess_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Tuple[dict, list]:
    score_pattern = r"^\d+(\.\d+)?/\d+(\.\d+)?$"
    df = df[df["review_score"].str.match(score_pattern)]
    normalized_scores = df["review_score"].apply(
        lambda x: float(x.split("/")[0]) / float(x.split("/")[1])
        if float(x.split("/")[1]) != 0
        else 0.0
    )

    inputs = tokenizer(
        df["review_content"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    labels = normalized_scores.apply(
        lambda x: 2 if x > 0.6 else (1 if 0.5 <= x <= 0.6 else 0)
    )
    return inputs, labels


def split_dataset(inputs: dict, labels: list) -> Tuple:
    (
        train_inputs,
        test_inputs,
        train_mask,
        test_mask,
        train_labels,
        test_labels,
    ) = train_test_split(
        inputs["input_ids"],
        inputs["attention_mask"],
        labels,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dict(input_ids=train_inputs, attention_mask=train_mask),
            train_labels,
        )
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dict(input_ids=test_inputs, attention_mask=test_mask),
            test_labels,
        )
    )

    return train_dataset, test_dataset


def train_model(
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> TFAutoModelForSequenceClassification:
    model = TFAutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=3, dropout=0.3, attention_dropout=0.3
    )
    batch_size = 32
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, weight_decay=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(
        train_dataset.shuffle(1000).batch(batch_size),
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=test_dataset.batch(batch_size),
        callbacks=[early_stopping_callback],
    )
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    df: pd.DataFrame = load_dataset(TRAINING_DATASET_PATH)
    inputs, labels = preprocess_dataset(df, tokenizer)
    train_dataset, test_dataset = split_dataset(inputs, labels)

    model = train_model(train_dataset, test_dataset)
    model.save_pretrained(MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
