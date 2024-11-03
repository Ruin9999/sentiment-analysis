
from datasets import load_dataset

def load_rotten_tomatoes_dataset():
    dataset = load_dataset("rotten_tomatoes")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']
    return train_dataset, validation_dataset, test_dataset
