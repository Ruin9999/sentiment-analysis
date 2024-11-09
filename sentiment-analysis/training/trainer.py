import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy
import json
import os

class SentimentClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, config_path=None):
        super(SentimentClassifier, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.save_hyperparameters()

    def forward(self, text):
        return self.model(text)

    def training_step(self, batch, batch_idx):
        text, labels = batch
        outputs = self(text)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)

        self.train_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_train_acc = self.train_acc.compute()
        self.log('avg_train_loss', avg_train_loss, prog_bar=True, logger=True)
        self.log('avg_train_acc', avg_train_acc, prog_bar=True, logger=True)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        self.train_losses.clear()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        text, labels = batch
        outputs = self(text)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(outputs, labels)

        self.val_losses.append(loss.item())
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        avg_val_acc = self.val_acc.compute()
        self.log('avg_val_loss', avg_val_loss, prog_bar=True, logger=True)
        self.log('avg_val_acc', avg_val_acc, prog_bar=True, logger=True)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        self.val_losses.clear()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        text, labels = batch
        outputs = self(text)
        loss = self.criterion(outputs, labels)
        acc = self.test_acc(outputs, labels)

        self.test_losses.append(loss.item())
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)

        return loss

    def on_test_epoch_end(self):
        avg_test_loss = sum(self.test_losses) / len(self.test_losses)
        avg_test_acc = self.test_acc.compute()
        self.log('avg_test_loss', avg_test_loss, prog_bar=True, logger=True)
        self.log('avg_test_acc', avg_test_acc, prog_bar=True, logger=True)
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        self.test_losses.clear()
        self.test_acc.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


