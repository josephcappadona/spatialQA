# -*- coding: utf-8 -*-
"""finetuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19dP7shHx2GtLxBbiqqBHMvUl5DlmnHlz
"""

#!pip -qq install transformers datasets gputil

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

dataset_name = 'snli'
train_dataset = load_dataset(dataset_name, split='train')
val_dataset = load_dataset(dataset_name, split='validation')

model_name = 'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
model.to(device);

import torch
from torch.utils.data import Dataset

def encode_data(batch, tokenizer, max_seq_length=128):
    zipped_ps_and_hs = list(zip(batch['premise'], batch['hypothesis']))
    encoded_data = tokenizer.batch_encode_plus(zipped_ps_and_hs,
                                               max_length=max_seq_length,
                                               padding='max_length',
                                               truncation='only_first',
                                               return_tensors='pt')
    return encoded_data['input_ids'], encoded_data['attention_mask']


def extract_labels(dataset):
    return [int(l) for l in dataset['label']]


class NLIDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        self.encoded_data = encode_data(dataframe,
                                        tokenizer,
                                        max_seq_length=max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        return {
            'input_ids': self.encoded_data[0][i],
            'attention_mask': self.encoded_data[1][i],
            'labels': self.label_list[i]
        }
    
    def collate_fn(self, batch):
        input_id_list = torch.stack([x['input_ids'] for x in batch])
        attention_mask_list = torch.stack([x['attention_mask'] for x in batch])
        label_list = torch.tensor([x['labels'] for x in batch])

        return [{'input_ids': input_id_list,
                 'attention_mask': attention_mask_list},
                label_list]

BATCH_SIZE = 32
max_sent_length=128

filtered_train_data = train_dataset.filter(lambda x: x['label'] != -1) # https://github.com/huggingface/datasets/issues/296
train_dataset = NLIDataset(filtered_train_data, tokenizer, max_sent_length)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.collate_fn,
                                           shuffle=True)

filtered_val_data = val_dataset.filter(lambda x: x['label'] != -1)
val_dataset = NLIDataset(filtered_val_data, tokenizer, max_sent_length)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=val_dataset.collate_fn,
                                           shuffle=True)

from transformers import AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def compute_metrics(eval_pred):

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    accuracy = accuracy_score(labels, preds)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds = model(
                input_ids=batch_x['input_ids'].to(device),
                attention_mask=batch_x['attention_mask'].to(device)
            ).logits
            all_preds.append(np.argmax(preds.detach().cpu().numpy(), axis=1))
            all_labels.append(batch_y)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    accuracy = (preds == labels).astype(int).sum() / preds.shape[0]
    return accuracy

from transformers import TrainingArguments
from torch import nn
from torch import optim

args = TrainingArguments(
    output_dir = './training_out/',
    num_train_epochs = 2,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    learning_rate = 1e-5,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

from tqdm import tqdm

train_loss_history = []
val_accuracy_history = []
best_val_accuracy = 0
n_no_improve = 0
early_stop_patience=2
NUM_EPOCHS=1
  
for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, miniters=10)):

        preds = model(
            input_ids=batch_x['input_ids'].to(device),
            attention_mask=batch_x['attention_mask'].to(device)
        ).logits
        loss = criterion(preds, batch_y.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())

    """
        5(2). TODO: Your code here
    """
    model.eval()
    val_accuracy = evaluate(model, val_loader, device)
    val_accuracy_history.append(val_accuracy)
    if val_accuracy > best_val_accuracy:
        n_no_improve = 0
        best_val_accuracy = val_accuracy
        torch.save(model, 'best_model.pt')
    else:
        n_no_improve += 1

    if n_no_improve > early_stop_patience:
        print("Stopping early...")
        break

print("Best validation accuracy is: ", best_val_accuracy)

import pandas as pd

ax = pd.Series(train_loss_history).plot()
ax.figure.savefig('train_loss.png')

ax = pd.Series(val_accuracy_history).plot()
ax.figure.savefig('val_acc.png')

import GPUtil
GPUtil.showUtilization()


