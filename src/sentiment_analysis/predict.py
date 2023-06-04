import argparse
from typing import List
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf

CLASS_LABELS = ["Negative", "Neutral", "Positive"]
MODEL_PATH = "./tuned_model"


def load_model(model_path: str) -> TFDistilBertForSequenceClassification:
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    return model


def preprocess_text(text: str) -> List[str]:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="tf",
        return_token_type_ids=False,
    )
    input_ids = inputs["input_ids"]
    return input_ids


def predict_sentiment(model: TFDistilBertForSequenceClassification, text: str) -> str:
    preprocessed_text = preprocess_text(text)
    predictions = model.predict(preprocessed_text)[0]
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = CLASS_LABELS[predicted_class]
    return predicted_label


def main(input_text: str):
    model = load_model(MODEL_PATH)

    sentiment = predict_sentiment(model, input_text)
    print(sentiment)
    return sentiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input text")
    args = parser.parse_args()

    main(args.input)
