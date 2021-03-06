from config import config
config = config()

from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#Model
import torch
from torch import nn

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair, kaggle=False):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        if kaggle==False:
            self.labels = [np.int32(i[label_idx]) for i in dataset]
        else:
            self.labels = [np.int32(-1) for _ in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class preprocess():
    def __init__(self, train_path, test_path, kaggle_path, use_all):
        device = torch.device("cuda:0")

        bertmodel, vocab = get_pytorch_kobert_model()
        self.model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

        dataset_train = nlp.data.TSVDataset(train_path, field_indices=[1,2], num_discard_samples=1)
        dataset_test = nlp.data.TSVDataset(test_path, field_indices=[1,2], num_discard_samples=1)
        dataset_kaggle = nlp.data.TSVDataset(kaggle_path, field_indices=[1], num_discard_samples=1, encoding='cp949')

        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        data_train = BERTDataset(dataset_train, 0, 1, tok, config.max_len, True, False)
        data_test = BERTDataset(dataset_test, 0, 1, tok, config.max_len, True, False)
        data_kaggle = BERTDataset(dataset_kaggle, 0, 1, tok, config.max_len, True, False, kaggle=True)

        self.train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=config.batch_size, num_workers=5)
        self.test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=config.batch_size, num_workers=5)
        self.kaggle_dataloader = torch.utils.data.DataLoader(data_kaggle, batch_size=1, num_workers=5)
        if use_all:
            dataset_all = nlp.data.TSVDataset(config.all_path, field_indices=[1,2], num_discard_samples=1)
            data_all =  BERTDataset(dataset_all, 0, 1, tok, config.max_len, True, False)
            self.all_dataloader = torch.utils.data.DataLoader(data_all, batch_size=config.batch_size, num_workers=5, shuffle=True)
