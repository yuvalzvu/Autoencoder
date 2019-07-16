#!/usr/bin/env python
# coding: utf-8

# # Autoencoder Project
# I developed a simple autoencoder (by "pytorch") to "test the performance of the bottleneck". 
# When the main idea was to examine the relationship between the dimension of the data and the bottleneck of the network.
# I created a synthetic data consisting of 100-sized vectors that were randomly drawn (number of vectors set the dimension number). 
# For each dimension I have created data that characterizes it by multiplying in any random alpha of each vector for each new vector in dimension.

# Main code

import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import statistics
import pandas as pd
from scipy.io import mmread
from sklearn.model_selection import train_test_split
import json


# Tools Imports
import matplotlib.pyplot as plt
import numpy as np
import json

def generate_layer_sizes(bottleneck, max_layer):
    if (max_layer < bottleneck):
        bottleneck, max_layer = max_layer, bottleneck
    bigLayer = int(max_layer)
    bigMidLayer = int(max_layer - ((max_layer - bottleneck) / 2))
    midLayer = int(bigMidLayer - ((bigMidLayer - bottleneck) / 2))
    midSmallLayer = int(midLayer - ((midLayer - bottleneck) / 2))
    smallLayer = int(bottleneck)
    return bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer

class Encoder(nn.Module):
    def __init__(self, bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(bigLayer, bigMidLayer), 
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(bigMidLayer, midLayer),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(midLayer, midSmallLayer),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(midSmallLayer, smallLayer),
            nn.ReLU(),
        )
        
       
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer):
        super(Decoder, self).__init__()
        self.layers = []
        self.fc1 = nn.Sequential(
            nn.Linear(smallLayer, midSmallLayer),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(midSmallLayer, midLayer),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(midLayer, bigMidLayer),
            nn.ReLU(),            
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(bigMidLayer, bigLayer),
            #nn.ReLU()
            nn.Sigmoid()
        )

        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x




from torch.utils.data import Dataset, DataLoader
COLUMN_NAMES = list(range(1, 101))


class YAEDataset(Dataset):
    def __init__(self, columns=COLUMN_NAMES, dimentions=2, vector_size=100, use_uniform=True, test_size=1000, scalar_max=1.0, scalar_min=0):
        ab = pd.DataFrame(np.random.random(size=(vector_size, dimentions)))
        vectors = np.array(ab)
        
        self.newData = pd.DataFrame(columns=columns)
        self.newData = vectors[:,0] * 1.0
        for dime in range(1, dimentions):
            self.newData = np.vstack([self.newData, vectors[:,dime] * 1.0])
        for i in range(dimentions, test_size):
            vector_n = [0.0] * vector_size
            for dime in range(dimentions):
                if use_uniform:
                    scalar = np.random.uniform(low=scalar_min, high=scalar_max)
                else:
                    scalar = (np.random.random() * vector_max) - vector_min
                vector_n += scalar * vectors[:,dime]
            sum_vector = np.asarray(vector_n)
            
            self.newData = np.vstack([self.newData, sum_vector])
    def __len__(self):
        return len(self.newData)
    
    def __getitem__(self, i):
        expression = self.newData[i]
        expression_tensor = torch.from_numpy(expression)
        
        label = self.newData[i]
        label_tensor = torch.from_numpy(label)
        return expression_tensor, label_tensor

def basic_run(dimentions, use_uniform, epoch_count, scalar_max, scalar_min, vector_max, vector_min, vector_size, batch_size, test_size, bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer, iteration):
    global global_saved_data
    encoder = Encoder(bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer)
    decoder = Decoder(bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer)

    # Optim - MSEloss
    from torch import optim
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    loss = nn.MSELoss()
    
    EPOCHS = epoch_count # num of train iteration

    train_errors = [] # save the errors throughout the training to visualize the training process once it is done.
    test_errors = [] # save the errors throughout the testing to visualize the training process once it is done.

    yae_dataset = YAEDataset(dimentions=dimentions, vector_size=vector_size, use_uniform=use_uniform, test_size=test_size, scalar_max=scalar_max, scalar_min=scalar_min)
    
    train_dataset, test_dataset = train_test_split(yae_dataset, test_size=0.2)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for _ in range(EPOCHS):
        for n_batch, (real_batch,_) in enumerate(train_data_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            temp = real_batch
            x_encoded = encoder(temp.float())
            y = decoder(x_encoded)



            error = loss(temp.float(), y.float())
            train_errors.append(error.item())
            error.backward()

            decoder_optimizer.step()
            encoder_optimizer.step()

    # Check the error on the training data
    plt.plot(train_errors)
    plt.savefig('./figures2/train_dims_%03d_bottleneck_%03d_iteration_%02d.png' % (dimentions, smallLayer, iteration + 1))
    plt.clf()
    with open('./figures2/train_dims_%03d_bottleneck_%03d_iteration_%02d.json' % (dimentions, smallLayer, iteration + 1), 'w', encoding='utf-8') as jsn:
        json.dump(train_errors, jsn)
    global_saved_data[bottleneck][dimentions][iteration]['train'] = statistics.mean(train_errors[-int(len(train_errors)/2):])
    
    for n_batch, (real_batch,_) in enumerate(test_data_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        temp = real_batch
        x_encoded = encoder(temp.float())
        y = decoder(x_encoded)



        error = loss(temp.float(), y.float())
        test_errors.append(error.item())
        error.backward()

        decoder_optimizer.step()
        encoder_optimizer.step()

    # Check the error on the test data
    plt.plot(test_errors)
    plt.savefig('./figures2/test_dims_%03d_bottleneck_%03d_iteration_%02d.png' % (dimentions, smallLayer, iteration + 1))
    plt.clf()
    with open('./figures2/test_dims_%03d_bottleneck_%03d_iteration_%02d.json' % (dimentions, smallLayer, iteration + 1), 'w', encoding='utf-8') as jsn:
        json.dump(test_errors, jsn)
    global_saved_data[bottleneck][dimentions][iteration]['test'] = statistics.mean(test_errors[-int(len(test_errors)/2):])

use_uniform = True
epoch_count = 50
scalar_max = 1.0
scalar_min = 0.0
vector_min = 0.0
vector_size = 100
batch_size = 100
test_size = 10000
number_of_iterations = 3
                            #      Dict[bot, Dict[dim, Dict[iter, Dict[train/test, mean_error]]]]
global_saved_data = dict()  # type Dict[int, Dict[int, Dict[int, Dict[str, float]]]]

bottle_neck_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100]
dimentions_sizes = [1, 2, 3, 5, 10, 20, 50, 100]
# bottle_neck_sizes = [1, 2, 3, 4, 5]
# dimentions_sizes = [1, 2, 3, 5]
fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')

for bottleneck in bottle_neck_sizes:
    global_saved_data[bottleneck] = dict()
    for dimentions in dimentions_sizes:
        global_saved_data[bottleneck][dimentions] = dict()
        bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer = generate_layer_sizes(bottleneck, 100)
        vector_max = 1.0 / dimentions
        for iteration in range(number_of_iterations):
            global_saved_data[bottleneck][dimentions][iteration] = dict()
            basic_run(dimentions, use_uniform, epoch_count, scalar_max, scalar_min, vector_max, vector_min, vector_size, batch_size, test_size, bigLayer, bigMidLayer, midLayer, midSmallLayer, smallLayer, iteration)
with open('./figures2/global_data.json', 'w', encoding='utf-8') as gd:
    json.dump(global_saved_data, gd)

# Making the results visual

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

X = np.asarray(bottle_neck_sizes)
Y = np.asarray(dimentions_sizes)
X, Y = np.meshgrid(X, Y)

# Z = np.sqrt(X + Y)
# Z = np.fromfunction(lambda x, y: (global_saved_data[x][y][0]['test'] + global_saved_data[x][y][1]['test'] + global_saved_data[x][y][2]['test']) / 3, (X, Y), dtype=int)
print(Z)
# Z = (global_saved_data[X][Y][0]['test'] + global_saved_data[X][Y][1]['test'] + global_saved_data[X][Y][2]['test']) / 3
# for x0, x1 in enumerate(X):
#     for y0, y1 in enumerate(Y):
#         Z[x0][y0] = (global_saved_data[x1][y1][0]['test'] + global_saved_data[x1][y1][1]['test'] + global_saved_data[x1][y1][2]['test']) / 3
# print(Z)

Z = np.sqrt(X + Y)
# Z = np.zeros((len(bottle_neck_sizes), len(dimentions_sizes)))
local_saved_data = list()
for x0, x1 in enumerate(bottle_neck_sizes):
    local_saved_data.append(list())
    for y0, y1 in enumerate(dimentions_sizes):
        local_saved_data[x0].append(0)
        for i in range(3):
#             print("dims: %d. bonk: %d. iter: %d. %f" % (x, y, i, global_saved_data[x][y][i]['test']))
            local_saved_data[x0][y0] += global_saved_data[x1][y1][i]['test']
        local_saved_data[x0][y0] /= 3
        Z[y0, x0] = local_saved_data[x0][y0]
# Z = np.asarray(local_saved_data)
fig = plt.figure(figsize=(18, 16), dpi=80)
ax = fig.gca(projection='3d')
ax.set_xlabel("Bottle Neck")
ax.set_ylabel("Dimentions")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

