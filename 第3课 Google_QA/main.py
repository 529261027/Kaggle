'''
NLP: BERT
1. 文章预测的结果是不同的
2. torch -> TF2.0

'''

import os
import numpy as np
import pandas as pd
from transformers import *
import tokenizers
import tensorflow as tf

pd.set_option('display.max_columns', None)
PATH = './data'
df_train = pd.read_csv(PATH+'/train.csv')
df_test = pd.read_csv(PATH+'./test.csv')
# print(df_train.head())
# print(df_train.columns)
'''
input:  'question_title', 'question_body', 'answer'
output: question_asker_intent_understanding:
'''
input_categorical = df_train.columns[[1,2,5]]
output_categorical = df_train.columns[11:]

'''
transformers:
1. TFBret ...
2. TF-model
'''
BERT_PATH = './input'
'''
定义tokenizer
1. BertTokenizer 和 BertWordPieceTokenizer 本质上是一致的
2. 导入路径如果是按照输入格式，那么到文件夹路径即可。否则要到文件具体的路径。
'''
tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'/bert-base-uncased-vocab.txt')
# tokenizer_bert = tokenizers.BertWordPieceTokenizer(BERT_PATH+'/bert-base-uncased-vocab.txt')
MAX_SEQUENCE_LENGTH = 512

'''
BERT:
ids: 索引
mask: self-attention
token_types_ids: 两个句子之间的标识位置

question_title: 问题是什么？
question_body: 输入问题短文
answer: 答案

question_title + ' ' + question_body -> BERT1
question_title + ' ' + answer -> BERT2
BERT1 + BERT2 -> concat
Dense -> result
'''
'''
TF2.0 数据处理
ids，mask，token_types_ids
'''
def conver_to_transformer_data(question_title, question_body, answer, tokenizer, max_length):
    q_input = tokenizer.encode_plus(question_title + ' ' + question_body, None, add_special_tokens=True, max_length=max_length, truncation=True)
    a_input = tokenizer.encode_plus(question_body + ' ' + answer, None, add_special_tokens=True, max_length=max_length, truncation=True)

    q_input_ids = q_input['input_ids']
    q_mask = q_input['attention_mask']
    q_token_type_ids = q_input['token_type_ids']
    padding = [0] * (max_length - len(q_input_ids))
    q_input_ids = q_input_ids + padding
    q_mask = q_mask + padding
    q_token_type_ids = q_token_type_ids + padding

    a_input_ids = a_input['input_ids']
    a_mask = a_input['attention_mask']
    a_token_type_ids = a_input['token_type_ids']
    padding = [0] * (max_length - len(a_input_ids))
    a_input_ids = a_input_ids + padding
    a_mask = a_mask + padding
    a_token_type_ids = a_token_type_ids + padding

    return q_input_ids, q_mask, q_token_type_ids, a_input_ids, a_mask, a_token_type_ids

tmp = conver_to_transformer_data(df_train['question_title'][0], df_train['question_body'][0],df_train['answer'][0],tokenizer,MAX_SEQUENCE_LENGTH)
from tqdm.autonotebook import tqdm
def compute_input_array(df, columns, tokenizer, max_length):
    q_input_ids_list, q_mask_list, q_token_type_ids_list = [], [], []
    a_input_ids_list, a_mask_list, a_token_type_ids_list = [], [], []
    for _, i in tqdm(df[columns].iterrows()):
        t, q, a = i.question_title, i.question_body, i.answer
        q_input_ids, q_mask, q_token_type_ids, a_input_ids, a_mask, a_token_type_ids = conver_to_transformer_data(t, q, a, tokenizer, max_length)

        q_input_ids_list.append(q_input_ids)
        q_mask_list.append(q_mask)
        q_token_type_ids_list.append(q_token_type_ids)

        a_input_ids_list.append(a_input_ids)
        a_mask_list.append(a_mask)
        a_token_type_ids_list.append(a_token_type_ids)

    return [np.array(q_input_ids_list,dtype=np.int32),
            np.array(q_mask_list,dtype=np.int32),
            np.array(q_token_type_ids_list,dtype=np.int32),
            np.array(a_input_ids_list,dtype=np.int32),
            np.array(a_mask_list,dtype=np.int32),
            np.array(a_token_type_ids_list,dtype=np.int32)]

# tmp = compute_input_array(df_train, input_categorical, tokenizer, MAX_SEQUENCE_LENGTH)

def model():
    q_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)
    a_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)

    q_mask = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)
    a_mask = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)

    q_token_type_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)
    a_token_type_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH),dtype=tf.int32)

    config = BertConfig.from_pretrained(BERT_PATH+'/bert-base-uncased-config.json')
    bert = TFBertModel.from_pretrained(BERT_PATH+'/bert-base-uncased-tf_model.h5', config=config)
    '''
    BERT:
    input: ids, mask, token_type_ids
    output: 单词的向量， CLS向量， output_hidden_states(BERT output_hidden_states=True)
    '''
    q_embedding = bert(q_ids,attention_mask=q_mask,token_type_ids=q_token_type_ids)[0]  # [batch, 512, 768]
    a_embedding = bert(a_ids,attention_mask=a_mask,token_type_ids=a_token_type_ids)[0]
    '''
    https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
    [batch, 512, 768] -> [batch, 768]
    '''
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)  # [batch, 768]
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)  # [batch, 768]

    concat = tf.keras.layers.Concatenate()([q,a])   # [batch, 768*2]
    x = tf.keras.layers.Dropout(0.2)(concat)
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)  # [batch, 30]
    model = tf.keras.Model(inputs=[q_ids,q_mask,q_token_type_ids,a_ids,a_mask,a_token_type_ids],outputs=x)
    return model

inputs = compute_input_array(df_train,input_categorical, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = np.array(df_train[output_categorical])

test_input = compute_input_array(df_test, input_categorical, tokenizer, MAX_SEQUENCE_LENGTH)

model = model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
'''
binary_crossentropy: 逻辑回归的损失函数，二分类
-( y * log(px) + (1-y) * log(1-px))
'''
model.compile(loss='binary_crossentropy',optimizer=optimizer)
model.fit(inputs,outputs,batch_size=2,epochs=3,verbose=1)
result = model.predict(test_input)


