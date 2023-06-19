import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import datetime
from tqdm import tqdm
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-05

config = {
  "lin": 64,
  "dropout": .5,
  "lr": 1e-5,
  "batch_size": 32)
}

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = (self.data.target>0).astype(int)
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.float),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def load_data(data_path, tokenizer, train_size=0.7, random_state=42, train_batch_size=32):
    df = pd.read_csv(data_path)

    new_df = df[['text', 'target']].copy()

    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)

    for ind in tqdm(new_df.index):
        new_df.loc[ind, 'text'] = new_df.loc[ind, 'text'].translate(translator)

    train_dataset=new_df.sample(frac=train_size, random_state=random_state)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': train_batch_size,
                  'shuffle': True,
                  'num_workers': 0
                  }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self, bert_model, lin_size=64, dropout=0.4):
        super(BERTClass, self).__init__()
        self.l1 = bert_model

        self.batch_norm_0 = torch.nn.BatchNorm1d(768)

        self.drop_1 = torch.nn.Dropout(dropout)
        self.dense_1 = torch.nn.Linear(768, lin_size)
        self.batch_norm_1 = torch.nn.BatchNorm1d(lin_size)

        self.drop_2 = torch.nn.Dropout(dropout)
        self.dense_2 = torch.nn.Linear(lin_size, 1)
        self.batch_norm_2 = torch.nn.BatchNorm1d(1)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        net = self.batch_norm_0(output_1)

        net = self.drop_1(net)
        net = self.dense_1(net)
        net = self.batch_norm_1(net)
        net = torch.nn.ReLU()(net)

        net = self.drop_2(net)
        net = self.dense_2(net)
        net = self.batch_norm_2(net)
        net = torch.nn.Sigmoid()(net)

        net = torch.flatten(net)

        return net

def loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets)

def validation(data_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().data.numpy().tolist())
            fin_outputs.extend(outputs.cpu().data.numpy().tolist())
    return fin_outputs, fin_targets

def train(config, data_path, bert_model, tokenizer, epochs=5):
    training_loader, testing_loader = load_data(data_path=data_path,tokenizer=tokenizer train_batch_size=config['batch_size'])

    model = BERTClass(bert_model, lin_size=config['lin'], dropout=config['dropout'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['lr'])

    for epoch in range(epochs):
        model.train()
        length = len(training_loader)
        for batch_id,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)

            out_cpu = (outputs.cpu().data.numpy()>0.5).astype(int)
            targets_cpu = targets.cpu().data.numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs_, targets = validation(testing_loader, model=model, device=device)

        outputs = (np.array(outputs_)>0.5).astype(int)

        val_accuracy = metrics.accuracy_score(targets, outputs)
        val_loss = loss_fn(outputs, targets)
    return model
