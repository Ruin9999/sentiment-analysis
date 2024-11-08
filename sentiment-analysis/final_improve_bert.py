# bert_sentiment_analysis.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configuration
CONFIG = {
    "model_name": "bert-base-uncased",
    "batch_size": 256,
    "max_length": 128,
    "epochs": 30,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
}

# Load Dataset
print("Loading dataset...")

dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# Preprocess data
print("Preprocessing data...")

tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

def preprocess_function(examples):

    max_length = CONFIG["max_length"]
    inputs = []

    for text in examples["text"]:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            tokens = tokens[: max_length // 2] + tokens[-(max_length // 2):]  # head+tail
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        inputs.append(input_ids)

    padded_inputs = tokenizer.pad(
        {"input_ids": inputs},
        padding="max_length",
        max_length=max_length
    )
    
    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": [[1] * len(input_ids) + [0] * (max_length - len(input_ids)) for input_ids in padded_inputs["input_ids"]]
    }


train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load BERT model for sequence classification
print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=2)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy=CONFIG["evaluation_strategy"],
    save_strategy=CONFIG["save_strategy"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["epochs"],
    weight_decay=CONFIG["weight_decay"],
    logging_dir='./logs',
    logging_steps=CONFIG["logging_steps"],
)


# Initialize Trainer
print("Training model...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test dataset
print("Evaluating model...")
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# Save the model
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
