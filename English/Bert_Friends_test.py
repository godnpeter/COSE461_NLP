#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://mccormickml.com/2019/07/22/BERT-fine-tuning/#2-loading-cola-dataset


# In[2]:


import json
#f= open('C:/Users/godnp/Downloads/COSE461_NLP-master/COSE461_NLP-master/English/EmotionLines/Friends/friends_train.json', "r")
#f= open('/home/hwang/EmotionLines/Friends/friends_train.json', "r")
f= open('./EmotionLines/Friends/friends_train.json', "r")


# In[3]:


data = json.load(f)


# In[4]:


#현재 json 파일이 보면 여러개의 dialect으로 구성되어 있음. 각 dialect에 접근하는데 data[i]이며 len(speech)를 통해 총 몇 번의 conversation이 이루어지는지 확인 가능
"""
for i in range(10):
    speech = data[i]
    print(len(speech))
"""

# In[5]:


speech = data[0]
speech[0]["utterance"]


# In[6]:


#어차피 데이터 내에 우리가 필요한것은 실제 말하는 전문, 그리고 emotion이다. 
#그러나 이 데이터셋에는 특이한 값, annotation이 존재한다.
#annotation은 여러가지 감정들에 대한 "정도"를 의미하는 것 같은데 우선 당장은 사용하지 않는 방향으로 생각해도 될 것 같다.
# 물론 성능이 부족하다 싶으면, 이 annotation까지 이용하는 방향으로 생각할 것
"""
print(len(data))
"""

# In[7]:


train_data, train_label = [], []

for i in range(len(data)):
    conversation = data[i]
    for j in range(len(conversation)):
        train_data += [conversation[j]["utterance"]]
        train_label += [conversation[j]["emotion"]]
                    


# In[8]:


#train_label = [neutral, joy, sadness, fear, anger, surprise, disgust] 이니 이거에 맞게 indexing 시켜줌

labels = []

for i in train_label:
    
    if i == 'neutral':
        num = 0
    elif i == 'joy':
        num = 1
    elif i == 'sadness':
        num = 2
    elif i == 'fear':
        num = 3
    elif i == 'anger':
        num = 4
    elif i == 'surprise':
        num = 5
    elif i == 'disgust':
        num = 6
    elif i == 'non-neutral':
        num = 7
    else:
        num = i
    
    labels.append(num)

len(labels)


# In[9]:


#f = open('C:/Users/godnp/Downloads/COSE461_NLP-master/COSE461_NLP-master/English/EmotionLines/Friends/friends_dev.json', "r")
#f= open('/home/hwang/EmotionLines/Friends/friends_dev.json', "r")
f= open('./EmotionLines/Friends/friends_dev.json', "r")

data = json.load(f)

val_data, val_label = [], []

for i in range(len(data)):
    conversation = data[i]
    for j in range(len(conversation)):
        val_data += [conversation[j]["utterance"]]
        val_label += [conversation[j]["emotion"]]

val_labels = []

for i in val_label:
    
    if i == 'neutral':
        num = 0
    elif i == 'joy':
        num = 1
    elif i == 'sadness':
        num = 2
    elif i == 'fear':
        num = 3
    elif i == 'anger':
        num = 4
    elif i == 'surprise':
        num = 5
    elif i == 'disgust':
        num = 6
    elif i == 'non-neutral':
        num = 7
    else:
        num = i
    
    val_labels.append(num)

        


# In[10]:


#f = open('C:/Users/godnp/Downloads/COSE461_NLP-master/COSE461_NLP-master/English/EmotionLines/Friends/friends_test.json', "r")
#f= open('/home/hwang/EmotionLines/Friends/friends_test.json', "r")
f= open('./EmotionLines/Friends/friends_test.json', "r")

data = json.load(f)

test_data, test_label = [], []

for i in range(len(data)):
    conversation = data[i]
    for j in range(len(conversation)):
        test_data += [conversation[j]["utterance"]]
        test_label += [conversation[j]["emotion"]]
        
        
test_labels = []

for i in test_label:
    
    if i == 'neutral':
        num = 0
    elif i == 'joy':
        num = 1
    elif i == 'sadness':
        num = 2
    elif i == 'fear':
        num = 3
    elif i == 'anger':
        num = 4
    elif i == 'surprise':
        num = 5
    elif i == 'disgust':
        num = 6
    elif i == 'non-neutral':
        num = 7
    else:
        num = i
    
    test_labels.append(num)


# In[11]:


#최종 Kaggle에 제출할 prediction dataset 전처리
#기존에 주어진 en_data.csv 파일을 online converter를 이용해 json 파일로 convert해서 전처리
# https://csvjson.com/csv2json
#f = open('C:/Users/godnp/Downloads/COSE461_NLP-master/COSE461_NLP-master/English/EmotionLines/Friends/csvjson.json', "r", encoding='utf-8')
#f= open('/home/hwang/EmotionLines/Friends/csvjson.json', "r", encoding='utf-8')
f= open('./EmotionLines/Friends/csvjson.json', "r", encoding='utf-8')


data = json.load(f)

fin_test_data, fin_test_labels = [], []

for j in range(len(data)):
    fin_test_data += [data[j]["utterance"]]
    fin_test_labels += [data[j]["id"]]


# In[ ]:





# In[12]:


#####################################DATA PREPROCESSING FINISHED##############################################################


# In[13]:


####################################MODEL TRAINING START###################################################


# In[14]:


#https://mccormickml.com/2019/07/22/BERT-fine-tuning/#2-loading-cola-dataset


# In[15]:


import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[16]:


# transformers module 사용

from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 8, output_attentions= False, output_hidden_states = False)
model.cuda()


# In[17]:


#Input data들을 tokenize 하고 각 token들을 각자의 words ID에 mapping 하기
input_ids = []
attention_masks = []

# For every sentence...
for sent in train_data:
    # `encode_plus` 는:
    #   (1) 문장 전체를 tokenize
    #   (2) 각 문장 앞에 [CLS] token 추가
    #   (3) 각 문장 마지막에 [SEP] token 추가
    #   (4) 각 token들을 각자의 ID에 map
    #   (5) max_length에 맞게 padding 혹은 truncate
    #   (6) [PAD] token들을 위해 attention mask 작성
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # encode 해야하는 문장
                        add_special_tokens = True, # '[CLS]' 랑 '[SEP]' 추가하기
                        max_length = 64,           # 모든 문장들을 Pad & truncate
                        pad_to_max_length = True,
                        return_attention_mask = True,   # attn. masks. 생성
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # encode된 문장들과 각 attention mask들을 list에 추가    
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

# lists들을 into tensors들로 변환
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# sentence 0 출력해보기
print('Original: ', train_data[0])
print('Token IDs:', input_ids[0])


# In[ ]:





# In[18]:


#Input data들을 tokenize 하고 각 token들을 각자의 words ID에 mapping 하기
val_input_ids = []
val_attention_masks = []

# For every sentence...
for sent in val_data:
    # `encode_plus` 는:
    #   (1) 문장 전체를 tokenize
    #   (2) 각 문장 앞에 [CLS] token 추가
    #   (3) 각 문장 마지막에 [SEP] token 추가
    #   (4) 각 token들을 각자의 ID에 map
    #   (5) max_length에 맞게 padding 혹은 truncate
    #   (6) [PAD] token들을 위해 attention mask 작성
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # encode 해야하는 문장
                        add_special_tokens = True, # '[CLS]' 랑 '[SEP]' 추가하기
                        max_length = 64,           # 모든 문장들을 Pad & truncate
                        pad_to_max_length = True,
                        return_attention_mask = True,   # attn. masks. 생성
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # encode된 문장들과 각 attention mask들을 list에 추가   
    val_input_ids.append(encoded_dict['input_ids'])
    val_attention_masks.append(encoded_dict['attention_mask'])

# lists들을 into tensors들로 변환
val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(val_labels)

# sentence 0 출력해보기
print('Original: ', val_data[0])
print('Token IDs:', val_input_ids[0])


# In[19]:


print(type(labels), labels.shape)


# In[20]:


test_input_ids = []
test_attention_masks = []

# For every sentence...
for sent in test_data:
    # `encode_plus` 는:
    #   (1) 문장 전체를 tokenize
    #   (2) 각 문장 앞에 [CLS] token 추가
    #   (3) 각 문장 마지막에 [SEP] token 추가
    #   (4) 각 token들을 각자의 ID에 map
    #   (5) max_length에 맞게 padding 혹은 truncate
    #   (6) [PAD] token들을 위해 attention mask 작성
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # encode 해야하는 문장
                        add_special_tokens = True, # '[CLS]' 랑 '[SEP]' 추가하기
                        max_length = 64,           # 모든 문장들을 Pad & truncate
                        pad_to_max_length = True,
                        return_attention_mask = True,   # attn. masks. 생성
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # encode된 문장들과 각 attention mask들을 list에 추가   
    test_input_ids.append(encoded_dict['input_ids'])
    
    test_attention_masks.append(encoded_dict['attention_mask'])

# lists들을 into tensors들로 변환
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_labels)

# sentence 0 출력해보기
print('Original: ', test_data[0])
print('Token IDs:', test_input_ids[0])


# In[21]:


fin_test_input_ids = []
fin_test_attention_masks = []

# For every sentence...
for num, sent in enumerate(fin_test_data):
    #blank sentence 처리하기
    if type(5) == type(sent):
        sent = 'error sentence'
    
    # `encode_plus` 는:
    #   (1) 문장 전체를 tokenize
    #   (2) 각 문장 앞에 [CLS] token 추가
    #   (3) 각 문장 마지막에 [SEP] token 추가
    #   (4) 각 token들을 각자의 ID에 map
    #   (5) max_length에 맞게 padding 혹은 truncate
    #   (6) [PAD] token들을 위해 attention mask 작성
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # encode 해야하는 문장
                        add_special_tokens = True, # '[CLS]' 랑 '[SEP]' 추가하기
                        max_length = 64,           # 모든 문장들을 Pad & truncate
                        pad_to_max_length = True,
                        return_attention_mask = True,   # attn. masks. 생성
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # encode된 문장들과 각 attention mask들을 list에 추가    
    fin_test_input_ids.append(encoded_dict['input_ids'])
    fin_test_attention_masks.append(encoded_dict['attention_mask'])

# lists들을 into tensors들로 변환
fin_test_input_ids = torch.cat(fin_test_input_ids, dim=0)
fin_test_attention_masks = torch.cat(fin_test_attention_masks, dim=0)
fin_test_labels = torch.tensor(fin_test_labels)

# sentence 0 출력해보기
print('Original: ', test_data[0])
print('Token IDs:', test_input_ids[0])


# In[22]:


len(fin_test_input_ids)


# In[23]:


params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# In[24]:


print(input_ids.shape)
print(len(input_ids))
print(len(torch.cat((input_ids, val_input_ids), 0)))
print(len(val_input_ids))
print(len(test_input_ids))
print(type(attention_masks))


# In[25]:


from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

train_dataset = TensorDataset(input_ids,attention_masks,labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
fin_test_dataset = TensorDataset(fin_test_input_ids, fin_test_attention_masks, fin_test_labels)


# DataLoader를 학습에 이용할 시, batch size를 정해줘야한다. 
# BERT을 Fine-Tuning 에 사용할 때, BERT 논문 작성자들은 batch size를 16 혹은 32를 권장했다.
#해당 프로젝트에서는 32를 사용했다.
batch_size = 32

#Training 이랑 Validation, 그리고 실제 final evaluation을 위한 데이터셋들의 DataLoader를 선언
train_dataloader = DataLoader(
            train_dataset,  # training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, #validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

test_dataloader = DataLoader(
            test_dataset, # test samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

fin_test_dataloader = DataLoader(
            fin_test_dataset, # Final prediction test set
            sampler = SequentialSampler(fin_test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# In[26]:


train_dataloader


# In[27]:


#optimizer 선언. 
#Learning rate 및 regularization term 선언

from transformers import AdamW
import os

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook has 2e-5
                  eps = 1e-4 # args.adam_epsilon  - default is 1e-8.
                )


# In[28]:



from transformers import get_linear_schedule_with_warmup

# Training epoch의 수. BERT 저자들은 2~4의 값을 추천함 
epochs = 2

total_steps = len(train_data) * epochs

# learning rate scheduler 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


# In[29]:


import numpy as np

# prediction vs labels 들의 accuracy를 측정하기 위한 function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[30]:


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    # Format = hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[31]:


test_input_ids.shape


# In[32]:


labels.shape


# In[33]:


#이제부터 모델 학습을 시작
#Model Training Start


import random
import numpy as np


# Seed value를 정해놔서 해당 결과의 재생산이 가능하도록 함.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#Training loss, validation loss, validation accuracy, 그리고 timing을 저장
training_stats = []

# 학습에 걸린 전체 시간을 저장
total_t0 = time.time()

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Training set 전체에 대해 한번 학습

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 1 Training epoch의 시간 재기
    t0 = time.time()

    # 해당 epoch의 total loss 초기화
    total_train_loss = 0
    
    # Model을 training mode로 선언해주기. 
    # training mode로 선언할 때랑, test mode로 선언할 때랑 dropout, batchnorm은 다르게 작동한다.
    model.train()

    # training data의 each batch에 대해 학습
    for step, batch in enumerate(train_dataloader):
        
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            
            # 현재 얼마나 진행됐는지 
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Dataloader에서 온 training batch unpacking
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Pytorch에서는 backward pass를 하기 전에 항상 이전 calculated gradient 값들을 초기화해줘야한다.
        model.zero_grad()        

        # forward pass 실행하기(training batch에 대해 model evaluation).
        # `model' function documentation은 해당 사이트에서 확인: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 우리의 model은 loss랑 logit을 반환한다
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # average loss 측정을 위한 training loss accumulation
        total_train_loss += loss.item()

        # gradient calculation을 위해 backwward pass
        loss.backward()

        # gradients의 norm을  1.0으로 clipping.
        # "exploding gradients" 문제의 발생을 예방
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 계산된 gradient를 기반으로 update parameters
        # 아까 선언한 optimizer가 parameter update 방식을 정의해줌
        optimizer.step()

        # learning rate을 update해주기
        scheduler.step()

    # 모든 batch들 대상으로 average loss 계산하기
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # 해당 epoch가 진행되는데 걸린 시간
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # 각 training epoch 후에, 우리의 모델의 performance를 validation set으로 평가

    print("")
    print("Running Validation...")

    t0 = time.time()

    # model의 performance 측정을 위해 evaluation mode로 선언
    model.eval()
 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # 1 epoch에 대해 evaluation
    for batch in validation_dataloader:
        
        # Dataloader에서 온 validation batch unpacking
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # pytorch에게 현재 evaluation mode이기 때문에 compute graph의 생성을 하지 않도록 하기
        with torch.no_grad():        

            # Forward pass를 통해 logit prediction들을 계산
            # `model' function documentation은 해당 사이트에서 확인: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 해당 model의 "logits" output은 score을 의미한다. 아직 softmax을 통과하기 전이다.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # validation loss 축적하기
        total_eval_loss += loss.item()

        # logits들과 labels들을 CPU로 옮기기
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 해당 test batch에 대한 accuracy를 계산 및 축적
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # 해당 validation run에 대한 최종 accuracy 출력
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # 모든 batch에 대한 average loss 계산
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # validation run이 얼마나 걸렸는지 측정
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # 해당 epoch의 statistics 기록
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


# In[34]:


# 여기서는 test_data 에 대한 evaluation을 진행해봤다.
    
model.eval()

total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

# 1 epoch에 대해 evaluation
for batch in test_dataloader:
        
    # Dataloader에서 온 test batch unpacking
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
        
    # pytorch에게 현재 evaluation mode이기 때문에 compute graph의 생성을 하지 않도록 하기
    with torch.no_grad():        

        # Forward pass를 통해 logit prediction들을 계산
        # `model' function documentation은 해당 사이트에서 확인: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 해당 model의 "logits" output은 score을 의미한다. 아직 softmax을 통과하기 전이다.
        (loss, logits) = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
    # validation loss 축적하기
    total_eval_loss += loss.item()

    # logits들과 labels들을 CPU로 옮기기
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 해당 test batch에 대한 accuracy를 계산 및 축적
    total_eval_accuracy += flat_accuracy(logits, label_ids)
        

# 해당 validation run에 대한 최종 accuracy 출력
avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
print("  Accuracy: {0:.2f}".format(avg_test_accuracy))

# 모든 batch에 대한 average loss 계산
avg_test_loss = total_eval_loss / len(test_dataloader)
    
# validation run이 얼마나 걸렸는지 측정
test_time = format_time(time.time() - t0)
    
print("  Test Loss: {0:.2f}".format(avg_test_loss))
print("  Test took: {:}".format(test_time))


# In[35]:


#모델 저장하는 코드
import os


output_dir = './model_save/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, 'module') else model  
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# In[36]:



#get_ipython().system('ls -l --block-size=K ./model_save/')


# In[37]:


# 이미 저장되어 있는, training을 마친 model loading 하기
import os


output_dir = './model_save/'

model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# 모델을 GPU에 복사하기
model.to(device)


# In[38]:


#학습한 모델을 이용해 en_data.csv의 prediction label 구하기
#이전에 해당 task를 위한 dataset을 fin_test_dataloader으로 변환 완료

fin_test_logits = []
i=0

# 1 epoch에 대해 evaluation
for batch in fin_test_dataloader:
        
    print(i)
    i= i+1
    
    # Dataloader에서 온 fin_test batch unpacking
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
        
    # pytorch에게 현재 evaluation mode이기 때문에 compute graph의 생성을 하지 않도록 하기
    with torch.no_grad():        

        # Forward pass를 통해 logit prediction들을 계산
        # `model' function documentation은 해당 사이트에서 확인: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 해당 model의 "logits" output은 score을 의미한다. 아직 softmax을 통과하기 전이다.
        (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            


    # logits들과 labels들을 CPU로 옮기기
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    
    fin_test_logits.append(logits)


# In[ ]:





# In[39]:


#test_logits
len(fin_test_logits)
list(np.argmax(fin_test_logits[86], axis=1))


# In[40]:


#모델을 통해 얻은 prediction label들의 ID 값을 각 영어 단어로 변환해주기 

final_test_logits=[]

for i in fin_test_logits:
    for j in list(np.argmax(i, axis=1)): 
        final_test_logits.append(j)
#print(final_test_logits)

final_labels = []

for i in final_test_logits:
    
    if i == 0:
        num = 'neutral'
    elif i == 1:
        num = 'joy'
    elif i == 2:
        num = 'sadness'
    elif i == 3:
        num = 'fear'
    elif i == 4:
        num = 'anger'
    elif i == 5:
        num = 'surprise'
    elif i == 6:
        num = 'disgust'
    elif i == 7:
        num = 'non-neutral'
    else:
        num = i
    
    final_labels.append([num])


# In[41]:


final_labels


# In[42]:


#최종 prediction label list를 csv 파일로 출력하기
import csv 

file = open('sentiment_result.csv', 'w+', newline ='') 
 
with file:	 
    write = csv.writer(file) 
    write.writerows(final_labels) 


# In[ ]:




