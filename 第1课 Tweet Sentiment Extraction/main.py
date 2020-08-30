'''
Pytorch
BERT

1. 数据分析，数据清洗
2. 构建DataLoader
3. 构建BERT模型
    3.1 transformers，导入预训练好的BERT模型
    3.2 BERT之后，接自己的训练模型
    3.3 构建LOSS
    3.4 构建optimizer，EarlyStopping
4. 训练
5. 测试
'''

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import transformers
import tokenizers
pd.set_option('display.max_columns', None)

path = './input'
df_train = pd.read_csv(os.path.join(path, 'train.csv'))
df_test = pd.read_csv(os.path.join(path, 'test.csv'))
print(df_train.info())
print(df_train.head())

df_train.dropna(inplace=True)

class config:
    MAX_LENTH = 128
    TRAIN_BATCH_SIZE = 2
    TEST_BATCH_SIZE = 32
    EPOCH = 3
    BERT_PATH = './input'
    MODEL_PATH = 'pytorch_model.bin'
    TOKENIZER = tokenizers.BertWordPieceTokenizer(os.path.join(BERT_PATH, 'vocab.txt'), lowercase=True)

# test:
tmp = config.TOKENIZER.encode('negative neutral positive')
print(tmp.ids)
'''
1. 构建DataLoader
transformers: https://huggingface.co/models 
BERT的tokenizer输出
1. ids
2. type_ids
3. tokens
4. offsets
'''
class TweetDataset:
    def __init__(self, tweet, selected_tweet, sentiment):
        self.tweet = tweet
        self.selected_text = selected_tweet
        self.sentiment = sentiment
    def __len__(self):
        return len(self.tweet)
    def __getitem__(self, item):
        tweet = self.tweet[item]
        selected_text = self.selected_text[item]
        sentiment = self.sentiment[item]
        self.tokenizer = config.TOKENIZER
        '''
        BERT模型需要的输入格式
        1) ids:text -> index
        2) mask: 参与到self-attention
        3) token_type_ids: 标识两个句子
        '''
        # 1）找到训练的标签：start，end
        idx0 = None
        idx1 = None
        for i, text in enumerate(tweet):
            if text == selected_text[0] and tweet[i:i + len(selected_text)] == selected_text:
                idx0 = i
                idx1 = i + len(selected_text) - 1

        tok_tweet = self.tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids[1:-1]
        tweet_offset = tok_tweet.offsets[1:-1]  # (0,0), (0,5)

        char_target = [0] * len(tweet)
        char_target[idx0:idx1+1] = [1] * len(selected_text)

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offset):
            if sum(char_target[offset1:offset2]) > 0:
                target_idx.append(j)
        target_start = target_idx[0]
        target_end = target_idx[-1]

        '''
        ids, mask, token_type_ids
        '''
        sentiment_id = {
            'negative':4997,
            'neutral':8699,
            'positive':3893
        }
        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig
        token_type_ids = [0, 0, 0] + [1] * (len(input_ids) - 3)
        mask = [1] * len(input_ids)
        tweet_offset = [(0,0)] * 3 + tweet_offset
        target_start += 3
        target_end += 3

        # padding, max_len < 128
        padding_lenght = config.MAX_LENTH - len(input_ids)
        if padding_lenght > 0:
            input_ids = input_ids + [0] * padding_lenght
            token_type_ids = token_type_ids + [0] * padding_lenght
            mask = mask + [0] * padding_lenght
            tweet_offset = tweet_offset + ([(0,0)] * (padding_lenght))

        return {
            'ids':torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long),
            'mask':torch.tensor(mask,dtype=torch.long),
            'tweet_off':torch.tensor(tweet_offset,dtype=torch.long),
            'target_start':torch.tensor(target_start, dtype=torch.long),
            'target_end':torch.tensor(target_end, dtype=torch.long),
            'tweet':tweet,
            'selected':selected_text
        }
'''
test
'''
tmp = TweetDataset(df_train['text'],df_train['selected_text'],df_train['sentiment'])
# data_tmp = tmp.__getitem__(0)
'''
BERT的输出：
https://www.cnblogs.com/deep-deep-learning/p/12792041.html
sequence_output, pooled_output, (hidden_states), (attentions)
1) sequence_output：输出的序列，所有单词的embedding [batch, length, embedding]
2) pooled_output: CLS的输出，[batch, embedding]
3) hidden_states: 输出BERT模型所有层的输出 13 * [batch, length, embedding]。(model_hidden_states=True)
4) attenions: 输出attentions
'''
class Tweet(transformers.BertPreTrainedModel):
    def __init__(self,conf):
        super(Tweet, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,config=conf)
        '''
        默认的方式就是可以训练的，这个我们只是说明一下。
        如果不训练BERT，False。
        训练的效果好与不训练
        '''
        for param in self.bert.parameters():
            param.requires_grad = True
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768*2, 2)
    def forward(self, ids, mask, token_type_ids):
        '''config : hidden_states = True'''
        _, _, output = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids)
        out = torch.cat((output[-1],output[-2]), dim=-1)
        out = self.drop_out(out)  # 768 * 2
        logist = self.l0(out)   # 768 * 2 -> 2
        start_logits, end_logits = logist.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, length]
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

# model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
# model_config.output_hidden_states = True
# tmp_model = Tweet(model_config)
# tmp_result = tmp_model(tmp.__getitem__(2)['ids'].reshape(1,-1),tmp.__getitem__(2)['mask'].reshape(1,-1),tmp.__getitem__(2)['token_type_ids'].reshape(1,-1))

'''
构建loss
'''
def loss_fn(start_logist, end_logist, start_position, end_position):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logist, start_position)
    end_loss = loss_fct(end_logist, end_position)
    return start_loss + end_loss

'''
构建optimizer
AdamW
'''
model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
model_config.output_hidden_states = True
model = Tweet(conf=model_config)

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimzer_parameter = [
    {'params':[p for n, p in param_optimizer if not any(i in n for i in no_decay)], 'weight_decay':0.01,'lr':3e-5},
    {'params':[p for n, p in param_optimizer if any(i in n for i in no_decay)],  'weight_decay':0.0,'lr':5e-5}
]
from transformers import AdamW
optimzer = AdamW(optimzer_parameter,lr=5e-5)
# 动态调整learning rate方式
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer,factor=0.1,patience=3,eps=1e-8)

'''
Early Stop
'''
from utils import EarlyStopping
es = EarlyStopping(patience=3,path='./output/checkpoint.pt')

'''
定义DataLoader
'''
from torch.utils.data import DataLoader
train_dataloader = DataLoader(TweetDataset(df_train['text'], df_train['selected_text'], df_train['sentiment']),batch_size=config.TRAIN_BATCH_SIZE)

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jaccard_score(tweet,orig_selected, start_logist, end_logist, offset):
    if start_logist > end_logist:
        start_logist = end_logist
    # offset (0,1), (1,9)
    logist_selected = tweet[offset[start_logist][0] : offset[end_logist][1]]
    return jaccard(orig_selected, logist_selected)

from tqdm.autonotebook import tqdm
def main():
    for i in range(config.EPOCH):
        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        losses = 0
        for i, data in enumerate(tk0):
            start_logist, end_logist = model(data['ids'], data['mask'], data['token_type_ids'])
            loss = loss_fn(start_logist, end_logist, data['target_start'], data['target_end'])
            losses += loss*len(data['ids'])
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            output_start = torch.argmax(start_logist, dim=-1)
            output_end = torch.argmax(end_logist, dim=-1)
            jaccards = []
            for p_i, tweet in enumerate(data['tweet']):
                jaccard_s = calculate_jaccard_score(tweet, data['selected'][p_i], output_start[p_i], output_end[p_i], data['tweet_off'][p_i])
                jaccards.append(jaccard_s)
            tk0.set_postfix({'loss':loss.item(),'jaccard':np.mean(jaccards)})
        scheduler.step(losses)
        es(losses, model)
        if es.early_stop:
            break

if __name__ == '__main__':
    main()