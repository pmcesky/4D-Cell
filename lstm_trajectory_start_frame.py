import numpy as np
import pandas as pd
import networkx as nx 
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import json
from pathlib import Path
import time
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
import random
import _utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler



with open( './cells_of_interest.json', 'r') as f:
   cells_of_interest = json.load(f)

with open('./cells_info.pickle', 'rb') as f:
    cells_info = pickle.load(f)

with open('./embryo_cells_info.pickle', 'rb') as f:
    embryo_cells_info = pickle.load(f)

embryos_for_test = ['WT-EMB05','WT-EMB12','WT-EMB19','WT-EMB26']
embryos_for_cross_validation = [embryo_name for embryo_name in embryo_cells_info if embryo_name not in embryos_for_test]



########################################################################################## Model ############################################################
class LSTM(nn.Module):
    def __init__(self, output_size = 334, num_rnn_layer = 1):
        super().__init__()
        self.flat = nn.Flatten() # flatten features
        self.unflat = nn.Unflatten(1, (50,3))  # trajectory input of shape (N, Lx3) to (N, L, 3), where N is batch_size, L=50 is seq_length
        # Hidden layer
        self.rnn = nn.LSTM(3, 3, num_layers=num_rnn_layer, batch_first=True) # (x,y,z)
        # final Output layer
        self.fc_out = nn.LazyLinear(output_size) # output layer

    def forward(self, x, x_extra = None):
        """_summary_

        Parameters
        ----------
        x : tensor
            trajectory features
        x_extra : tensor
            extra features, like start_frame, lifespan, division orientations, by default None

        Returns
        -------
        tensor
            forward pass of (x, x_extra)
        """     
        # trajectory branch
        # convert x from (N, Lx3) to (N, L, 3), where L should be 50
        out = self.unflat(x)
        out, _ = self.rnn(out) # out is of shape (N, L, H_out), where N is batch_size, L is seq_length, H_out is hidden feature size. In our case, out is of shape (N, 50, 3)  
        out = self.flat(out) # to shape (N, 50*3)
        # no extra features as input, only trajectory
        if x_extra is None: 
            out = self.fc_out(out)
        else:
            x_extra = self.flat(x_extra)
            out = torch.cat((out, x_extra), dim = 1)
            out = self.fc_out(out)
        return out



#############################################################################################################
device = 'cuda' if torch.cuda.is_available else 'cpu'

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device=='cuda':
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


batch_size = 128 # Batch_size
num_epochs = 3000
test_interval = 10 # compute test accuracy on test_dl every some epochs
save_interval = 100 # save model every some epochs
lr = 0.001 
gamma = 0.999 # lr scheduler exp decay gamma, lr decay by gamma every epoch 
weight_decay=0.2 # L2 regularization
##############################################################################################################



########################################## lstm use trajectory and start_frame features ###########################################################################################
cv_train_loss = []
cv_train_accuracy = []
cv_val_loss = []
cv_val_accuracy = []
cv_test_accuracy = []
# Cross Validation
for i in range(6):
    print(f'{i}-fold')
    # train and val data
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = False, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = False, flatten = True)
    # test data
    X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = False, flatten = True)


    # # feature normalization
    # scaler = StandardScaler()
    # scaler.fit(np.array(X_train))
    # X_train = scaler.transform(np.array(X_train))
    # # standardize X_val and X_test
    # X_val = scaler.transform(np.array(X_val))
    # X_test = scaler.transform(np.array(X_test))

    # Trajectory + start_frame features
    X_train = np.array(X_train)[:,:151]
    X_val = np.array(X_val)[:,:151]
    X_test = np.array(X_test)[:,:151]

    # Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    # Dataloader 
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ### Train and Validate model
    model = LSTM(output_size=len(cells_of_interest)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    test_accuracy = []

    start_time = time.time()
    # Training
    for epoch in range(1,num_epochs+1):
        acc_train, loss_train = _utilities.train_rnn(model, train_dl, optimizer, device, gradient_clip=True)
        acc_valid, loss_valid = _utilities.evaluate_rnn(model, val_dl, optimizer, device)
        scheduler.step() # adjust lr
        train_loss.append(loss_train)
        train_accuracy.append(acc_train)
        val_loss.append(loss_valid)
        val_accuracy.append(acc_valid)
        print(f'Epoch: {epoch:04d}/{num_epochs:04d} | '
            f'Accuracy: {acc_train:.4f} | '
            f'Val_accuracy: {acc_valid:.4f} | '
            f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
        
        if epoch%test_interval == 0: # test accuracy
            # Test
            acc_test, _ = _utilities.evaluate_rnn(model, test_dl, optimizer, device)
            test_accuracy.append(acc_test)
            print(f'Test_accuracy: {acc_test:.4f}')
        # if epoch%save_interval == 0: # save test model
        #     torch.save(model.state_dict(), f"./lstm/model_pt/lstm_traj_sf_{i}_fold_CV_{epoch}_wd02.pt")
 
    # Final Test Accuracy
    acc_test, _ = _utilities.evaluate_rnn(model, test_dl, optimizer, device)
    # test_accuracy.append(acc_test)
    print(f'Final Test_accuracy: {acc_test:.4f}')
    # record for this CV fold
    cv_train_loss.append(train_loss[:])
    cv_val_loss.append(val_loss[:])
    cv_train_accuracy.append(train_accuracy[:])
    cv_val_accuracy.append(val_accuracy[:])
    cv_test_accuracy.append(test_accuracy[:])

with open( './lstm/lstm_traj_sf_cv_train_loss_wd02.json', 'w') as f:
   json.dump(cv_train_loss, f)
with open( './lstm/lstm_traj_sf_cv_val_loss_wd02.json', 'w') as f:
   json.dump(cv_val_loss, f)
with open( './lstm/lstm_traj_sf_cv_train_accuracy_wd02.json', 'w') as f:
   json.dump(cv_train_accuracy, f)
with open( './lstm/lstm_traj_sf_cv_val_accuracy_wd02.json', 'w') as f:
   json.dump(cv_val_accuracy, f)
with open( './lstm/lstm_traj_sf_cv_test_accuracy_wd02.json', 'w') as f:
   json.dump(cv_test_accuracy, f)



################################ Train on all cross_validation dataset and test on test set
lstm_train_loss = []
lstm_train_accuracy = []
lstm_test_accuracy = []
# train and test data
X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_cross_validation, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = False, flatten = True)
# test data
X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = False, flatten = True)


# # feature normalization
# scaler = StandardScaler()
# scaler.fit(np.array(X_train))
# X_train = scaler.transform(np.array(X_train))
# # standardize X_test
# X_test = scaler.transform(np.array(X_test))

# Trajectory + start_frame features
X_train = np.array(X_train)[:,:151]
X_test = np.array(X_test)[:,:151]


# Dataset
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
# Dataloader 
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


### Train and Test model
model = LSTM(output_size=len(cells_of_interest)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


start_time = time.time()
# Training
for epoch in range(1,num_epochs+1):
    acc_train, loss_train = _utilities.train_rnn(model, train_dl, optimizer, device, gradient_clip=True)
    scheduler.step() # adjust lr
    lstm_train_loss.append(loss_train)
    lstm_train_accuracy.append(acc_train)
    print(f'Epoch: {epoch:04d}/{num_epochs:04d} | '
        f'Accuracy: {acc_train:.4f} | '
        f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    
    if epoch%test_interval == 0: # test accuracy
        # Test
        acc_test, _ = _utilities.evaluate_rnn(model, test_dl, optimizer, device)
        lstm_test_accuracy.append(acc_test)
        print(f'Test_accuracy: {acc_test:.4f}')
    if epoch%save_interval == 0: # save model test
        torch.save(model.state_dict(), f"./lstm/model_pt/lstm_traj_sf_test_{epoch}_wd02.pt") 
# Test
acc_test, _ = _utilities.evaluate_rnn(model, test_dl, optimizer, device)
# test_accuracy.append(acc_test)
print(f'Final Test_accuracy: {acc_test:.4f}')

with open( './lstm/lstm_traj_sf_test_train_loss_wd02.json', 'w') as f:
   json.dump(lstm_train_loss, f)
with open( './lstm/lstm_traj_sf_test_train_accuracy_wd02.json', 'w') as f:
   json.dump(lstm_train_accuracy, f)
with open( './lstm/lstm_traj_sf_test_test_accuracy_wd02.json', 'w') as f:
   json.dump(lstm_test_accuracy, f)