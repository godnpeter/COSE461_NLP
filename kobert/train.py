import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
#from transformers.optimization import WarmupLinearSchedule

##Change
from config import config
from preprocess import preprocess
import csv
import argparse
##Change

parser = argparse.ArgumentParser()
#For Use All_data
parser.add_argument('--use_all', type=int, default=0)
#
args = parser.parse_args()
print args

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

config = config()

print("Start Preprocessing")
p = preprocess(config.train_path, config.test_path, config.kaggle_path)

device = torch.device("cuda:0")
model = p.model
if args.use_all:
    train_dataloader = p.all_dataloader
else:
    train_dataloader = p.train_dataloader
test_dataloader = p.test_dataloader
kaggle_dataloader = p.kaggle_dataloader
print("Finish Preprocessing")

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * config.num_epochs
warmup_step = int(t_total * config.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

print("Start Training")
from datetime import datetime

best_epoch=0
best_accuracy=-1
warmup_epoch = 5

for e in range(config.num_epochs):
    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (e, str(start)))
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % config.log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e, train_acc / (batch_id+1)))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    test_acc = test_acc / (batch_id+1)
    print("epoch {} test acc {}".format(e, test_acc))

    if best_accuracy < test_acc and e >= warmup_epoch:
        best_epoch = e
        best_accuracy = test_acc
        #Model save
        torch.save(model.state_dict(), "./model/model.epoch-" + str(e) + "_"+str(int(best_accuracy*10000)))
        #make pred
        wf = open("./result/sample_epoch_"+str(e)+"_"+str(int(best_accuracy*10000))+".csv",'w', encoding='euc-kr')
        wr=csv.writer(wf)
        wr.writerow(["Id", "Predicted"])
        for batch_id, (token_ids, valid_length, segment_ids,_) in enumerate(kaggle_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            out = model(token_ids, valid_length, segment_ids)
            _, max_indices = torch.max(out,1)
            wr.writerow([batch_id,max_indices.item()])
        wf.close()
