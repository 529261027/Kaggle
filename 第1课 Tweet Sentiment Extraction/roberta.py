'''
RoBERTa
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers
import tokenizers
import os
from transformers import AdamW

'''
https://huggingface.co/transformers/model_doc/roberta.html
https://huggingface.co/roberta-base#
'''
class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 2
    EPOCH = 5
    BERT_PATH = './roberta_input'
    SAVE_PATH = './output'
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=os.path.join(BERT_PATH, 'vocab.json'),
        merges_file=os.path.join(BERT_PATH,'merges.txt'),
        lowercase=True,
        # add_prefix_space=True
    )

# print(config.TOKENIZER.encode('hello world').ids)
# print(config.TOKENIZER.encode(' "<s>"').ids)

path = './input'
df_train = pd.read_csv(os.path.join(path, 'train.csv'))
df_test = pd.read_csv(os.path.join(path, 'test.csv'))

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

        tweet = ' ' + ' '.join(tweet.split())  # split()   去除掉空格和\n
        selected_text = ' ' + ' '.join(selected_text.split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for i, text in enumerate(tweet):
            if text == selected_text[1] and tweet[i:i + len_st] == selected_text[1:]:
                idx0 = i
                idx1 = i + len(selected_text) - 1
                break
        char_targets = [0] * len(tweet)
        char_targets[idx0:idx1+1] = [1]*len(selected_text)

        tok_tweet = self.tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids[1:-1]  # cls sep
        tweet_offset = tok_tweet.offsets[1:-1]

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offset):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)
        target_start = target_idx[0]
        target_end = target_idx[-1]

        sentiment_id = {
            'negative': 2430,
            'neutral': 7974,
            'positive': 1313
        }
        '''
        bos_token="<s>",  CLS
        eos_token="</s>", SEP
        sep_token="</s>", SEP
        
         - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``
        
        <s>
        '''
        input_ids = [0] + [sentiment_id[sentiment]] + [2,2] + input_ids_orig + [2]
        token_type_ids = [0] * len(input_ids)
        mask = [1] * len(input_ids)
        tweet_offset = [(0,0)] * 4 + tweet_offset
        target_start += 4
        target_end += 4
        padding_lenth = config.MAX_LEN - len(input_ids)
        if padding_lenth > 0:
            input_ids = input_ids + [0] * padding_lenth
            token_type_ids = token_type_ids + [0] * padding_lenth
            mask = mask + [0] * padding_lenth
            tweet_offset = tweet_offset + [(0,0)] * padding_lenth
        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tweet_off': torch.tensor(tweet_offset, dtype=torch.long),
            'target_start': torch.tensor(target_start, dtype=torch.long),
            'target_end': torch.tensor(target_end, dtype=torch.long),
            'tweet': tweet,
            'selected': selected_text
        }

t = TweetDataset(tweet=df_train['text'],selected_tweet=df_train['selected_text'],sentiment=df_train['sentiment'])
print(t.__getitem__(0))

class Tweet(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(Tweet, self).__init__(conf)
        self.bert = transformers.RobertaModel.from_pretrained(config.BERT_PATH, config=conf)
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
es = EarlyStopping(patience=3,path='./output/roberta_checkpoint.pt')

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