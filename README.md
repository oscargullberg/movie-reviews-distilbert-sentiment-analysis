# Movie review sentiment analysis using DistilBERT and tensorflow

Example of how to finetune a transformer model for sentiment analysis using tensorflow.  
Apple silicon GPU acceleration support from [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/).

- [Rotten Tomatoes movies and critic reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- [distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert)

## Requirements

- Python 3.11
- poetry

## Installation

`poetry install`

## Usage

### Train

`poetry run python src/sentiment_analysis/train.py`

### Predict

`poetry run python src/sentiment_analysis/predict.py --input "Pretty ok"`

> `"Neutral"`
