import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
import keras   # keras 2.0.6 tf 1.X   tf 2.X from tensorflow import keras

train_path = './input/training'
test_path = './input/test'
SIZE = 10

train_dir = os.listdir(train_path)
test_dir = os.listdir(test_path)

with open(os.path.join(train_path, train_dir[1])) as f:
    task = json.load(f)
# print(task)
# print(task.keys())

def plot_task(task):
    index = 0
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    task_num = len(task['train']) + len(task['test'])
    fig, ax = plt.subplots(2, task_num, figsize=(8,8))
    for k, v in task.items():
        for j in v:
            ax[0, index].imshow(j['input'], cmap=cmap, norm=norm)
            ax[0, index].set_xticklabels([])
            ax[0, index].set_yticklabels([])
            ax[0, index].tick_params(length=0)
            ax[0, index].set_title(k + ' Input')
            ax[1, index].imshow(j['output'], cmap=cmap, norm=norm)
            ax[1, index].set_xticklabels([])
            ax[1, index].set_yticklabels([])
            ax[1, index].tick_params(length=0)
            ax[1, index].set_title(k + ' Output')
            index += 1
    plt.tight_layout()
    plt.show()
# plot_task(task)

X_test, X_train, y_train = [], [], []
for file in test_dir:
    with open(os.path.join(test_path, file)) as f:
        task = json.load(f)
    xs_test, xs_train, ys_train = [], [], []
    for pair in task['train']:
        xs_train.append(pair['input'])
        ys_train.append(pair['output'])
    for pair in task['test']:
        xs_test.append(pair['input'])
    X_test.append(xs_test)
    X_train.append(xs_train)
    y_train.append(ys_train)

# for i in range(10):
#     print([np.array(j).shape for j in X_test[i]])
#     print([np.array(j).shape for j in X_train[i]])
#     print([np.array(j).shape for j in y_train[i]])
def get_new_matrix(x, y):  # [[1,2],[1,2]] -> [[1,2]]
    if len(set([np.array(i).shape for i in x])) > 1 or len(set([np.array(i).shape for i in y])) >1 :
        return [x[0]], [y[0]]
    else:
        return x,y

def repeat(x):
    '''
    [[1,2],[3,4]]
    :param x:
    :return:
    '''
    return np.concatenate([x]*(SIZE//len(x) + 1))[:SIZE]

def replace_values(x, dictionary):
    '''
    [[1,2,3], [3,4,5]]
    dictionary: [1,2,3,4,5,6]
    :param x:
    :param dictionary:
    :return:
    '''
    return np.array([dictionary[j] for i in x for j in i]).reshape(np.array(x).shape)

def get_outp(x):
    '''
    [[0,1,0,0..],
    [0,0,1,0,0...]]
    [0,1,0...  0,0,1....]
    :param x:
    :return:
    '''
    return keras.utils.to_categorical(x.flatten(), num_classes=10).flatten()

class ARCDataset(Dataset):
    def __init__(self,X, Y):
        self.x = X
        self.y = Y
        # 1. 进行数据shape的定义
        self.x, self.y = get_new_matrix(self.x, self.y)
        # 2. 复制数据
        self.x, self.y = repeat(self.x), repeat(self.y)
    def __len__(self):
        return SIZE
    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        rep_dic = np.arange(10)
        org_dic = np.arange(10)
        np.random.shuffle(rep_dic)
        dictionary = dict(zip(org_dic, rep_dic))
        # 数据进行转换，数据增强
        x, y = replace_values(x, dictionary), replace_values(y, dictionary)
        '''
        y = [[1,2,3], [4,5,6]]
        [1,2,3,4,5,6]
        [0,1,0,0,0...  0,0,1,0,0... ...]
        => 2 * 3 * 10
        '''
        outpr = get_outp(y)
        return x, outpr

# dataset = ARCDataset(X_train[0], y_train[0])
# tmp = dataset.__getitem__(0)
# data_loader = DataLoader(dataset, batch_size=128)
# for i in data_loader:
#     print(i[0].shape)
#     print(i[1].shape)

class BasicCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicCNN, self).__init__()
        self.conv_in = 3
        self.conv_out1 = 50
        self.conv_out2 = 100
        if input_dim[0] or input_dim[1] < 5:
            KERNEL_SIZE = 1
        else:
            KERNEL_SIZE = 3
        self.conv1 = nn.Conv2d(self.conv_in, self.conv_out1, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(self.conv_out1, self.conv_out2, kernel_size=KERNEL_SIZE)
        self.dense1 = nn.Linear(self.conv_out2, output_dim[0] * output_dim[1] * 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        '''
        [[[1,2,1], [2,3,4]]]
        [batch, imgx, imgy] => [batch, channel, imgx, imgy]
        :param x:
        :return:
        '''
        x = torch.cat([x.unsqueeze(0)]*3)  # [3, batch, imgx, imgy]
        x = x.permute((1,0,2,3)).float()
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))  # [batch, channel, imgx, imgy]
        pooling, _ = torch.max(conv2, dim=-1) # [batch, channel, imgx]
        pooling, _ = torch.max(pooling, dim=-1)  # [batch ,channel]
        result = self.dense1(pooling)  # [batch, channels]
        '''
        [batch, channels]
        [batch, -1, 10]
        [batch, channels]
        '''
        result = torch.softmax(result.reshape(result.shape[0],-1,10), dim=-1).reshape(result.shape[0], -1)
        return result
import cv2
def resize(x, input_dim, test_dim):
    if input_dim == test_dim:
        return x
    else:
        return cv2.resize(np.float32(x), input_dim, interpolation=cv2.INTER_AREA)


EPOCHS = 1
idx = 0
test_predictions = []
for x_, y_ in zip(X_train, y_train):
    dataset = ARCDataset(x_, y_)
    data_loader = DataLoader(dataset, batch_size=128)
    input_dim = np.array(x_[0]).shape
    output_dim = np.array(y_[0]).shape
    network = BasicCNN(input_dim, output_dim)
    optimizer = torch.optim.Adam(network.parameters(),lr=0.01)
    for epoch in range(EPOCHS):
        for i in data_loader:
            train_x, train_y = i
            logists = network(train_x)
            loss = nn.MSELoss()(train_y, logists)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(epoch, loss.item())
    for test_ in X_test[idx]:
        # x_test.shape <> input_dim
        test_ = resize(test_, input_dim, np.array(test_).shape)
        # input:[batch, input_dim] # output:[batch, seq]
        logist_test = network(torch.tensor(test_).unsqueeze(0)).detach().numpy()
        '''
        [-1, 10] -> argmax -> [-1,1] -> reshape(output_dim)
        '''
        logist_test = np.argmax(logist_test.reshape(-1, 10), axis=-1).reshape(output_dim)
        # print(logist_test.shape)
        # 作者：有一步reshape的操作。
        '''
        input_dim , output_dim
        test_input_dim, -> output_dim
        '''
        test_predictions.append(logist_test)
    idx += 1

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

results = []
for i in test_predictions:
    results.append(flattener([list(j) for j in i]))
print(results)



