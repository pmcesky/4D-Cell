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
import mlflow
from sklearn.preprocessing import StandardScaler


with open( './cells_of_interest.json', 'r') as f:
   cells_of_interest = json.load(f)

with open('./cells_info.pickle', 'rb') as f:
    cells_info = pickle.load(f)

with open('./embryo_cells_info.pickle', 'rb') as f:
    embryo_cells_info = pickle.load(f)

embryos_for_test = ['WT-EMB05','WT-EMB12','WT-EMB19','WT-EMB26']
embryos_for_cross_validation = [embryo_name for embryo_name in embryo_cells_info if embryo_name not in embryos_for_test]



# Model
class MLP(nn.Module):
    def __init__(self, output_size = 334, activation = nn.ReLU()):
        super().__init__()
        self.flat = nn.Flatten() # flatten features
        # Hidden layer
        self.act = activation # activation function for hidden layers, default is tanh
        self.fc1 = nn.LazyLinear(200) # process trajectory, 200 to 200
        self.fc2 = nn.LazyLinear(200) # output of trajectory
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
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
        identity = x.clone()
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout1(out)
        # add residual connection
        out = out + identity
        identity = out.clone()
        out = self.fc2(out)
        out = self.act(out)
        out = self.dropout2(out)
        out = out + identity
        # no extra features as input, only trajectory
        if x_extra is None: 
            out = self.fc_out(out)
        else:
            out = self.flat(out)
            x_extra = self.flat(x_extra)
            out = torch.cat((out, x_extra), dim = 1)
            out = self.fc_out(out)
        return out


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

# Batch_size
batch_size = 128

########################################## MLP use all features ###########################################################################################
# cv_train_loss = []
# cv_val_loss = []
# cv_val_accuracy = []
# # Cross Validation
# for i in range(6):
#     print(f'{i}-fold')
#     # train and val data
#     embryos_val = embryos_for_cross_validation[4*i:4*i+4]
#     embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
#     X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
#     X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
#     # test data
#     X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)


#     # feature normalization
#     scaler = StandardScaler()
#     scaler.fit(np.array(X_train))
#     X_train = scaler.transform(np.array(X_train))
#     # standardize X_val and X_test
#     X_val = scaler.transform(np.array(X_val))
#     X_test = scaler.transform(np.array(X_test))


#     # Dataset
#     train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
#     val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
#     test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
#     # Dataloader 
#     train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#     ### Train and Validate model
#     model = MLP(output_size=len(cells_of_interest)).to(device)
#     lr = 0.001
#     weight_decay=0.1 # L2 regularization
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

#     epochs = 5000
#     log_interval = 100 # per batch
#     save_interval = 500 # per epoch
#     train_loss = []
#     val_loss = []
#     val_accuracy = []

#     for epoch in range(1, epochs + 1):
#         print(f"Train epoch {epoch}, \t time {time.strftime('%H:%M:%S', time.localtime())}")
#         loss_train = _utilities.train(model, train_dl, optimizer, log_interval, device)
#         loss_val, accuracy_val = _utilities.test(model, val_dl, device)
#         train_loss.append(loss_train)
#         val_loss.append(loss_val)
#         val_accuracy.append(accuracy_val)
#         # we can save the model after each epoch
#         if epoch%save_interval == 0:
#             torch.save(model.state_dict(), f"./mlp/model_pt/mlp_{i}_fold_CV_{epoch}.pt")
#     # record for this CV fold
#     cv_train_loss.append(train_loss[:])
#     cv_val_loss.append(val_loss[:])
#     cv_val_accuracy.append(val_accuracy[:])

# with open( './mlp/cv_train_loss.json', 'w') as f:
#    json.dump(cv_train_loss, f)
# with open( './mlp/cv_val_loss.json', 'w') as f:
#    json.dump(cv_val_loss, f)
# with open( './mlp/cv_val_accuracy.json', 'w') as f:
#    json.dump(cv_val_accuracy, f)



########################################## MLP use only trajectory ###########################################################################################
cv_train_loss = []
cv_val_loss = []
cv_val_accuracy = []
# Cross Validation
for i in range(6):
    print(f'{i}-fold')
    # train and val data
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    # test data
    X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)


    # feature normalization
    scaler = StandardScaler()
    scaler.fit(np.array(X_train))
    X_train = scaler.transform(np.array(X_train))
    # standardize X_val and X_test
    X_val = scaler.transform(np.array(X_val))
    X_test = scaler.transform(np.array(X_test))

    # Trajectory only features
    X_train = X_train[:,:200]
    X_val = X_val[:,:200]
    X_test = X_test[:,:200]

    # Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    # Dataloader 
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ### Train and Validate model
    model = MLP(output_size=len(cells_of_interest)).to(device)
    lr = 0.001
    weight_decay=0.1 # L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    epochs = 5000
    log_interval = 100 # per batch
    save_interval = 500 # per epoch
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(1, epochs + 1):
        print(f"Train epoch {epoch}, \t time {time.strftime('%H:%M:%S', time.localtime())}")
        loss_train = _utilities.train(model, train_dl, optimizer, log_interval, device)
        loss_val, accuracy_val = _utilities.test(model, val_dl, device)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_accuracy.append(accuracy_val)
        # we can save the model after each epoch
        if epoch%save_interval == 0:
            torch.save(model.state_dict(), f"./mlp/model_pt/mlp_trajectory_{i}_fold_CV_{epoch}.pt")
    # record for this CV fold
    cv_train_loss.append(train_loss[:])
    cv_val_loss.append(val_loss[:])
    cv_val_accuracy.append(val_accuracy[:])

with open( './mlp/cv_train_loss_trajectory.json', 'w') as f:
   json.dump(cv_train_loss, f)
with open( './mlp/cv_val_loss_trajectory.json', 'w') as f:
   json.dump(cv_val_loss, f)
with open( './mlp/cv_val_accuracy_trajectory.json', 'w') as f:
   json.dump(cv_val_accuracy, f)

########################################## MLP use only trajectory + start_frame ###########################################################################################
cv_train_loss = []
cv_val_loss = []
cv_val_accuracy = []
# Cross Validation
for i in range(6):
    print(f'{i}-fold')
    # train and val data
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    # test data
    X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)


    # feature normalization
    scaler = StandardScaler()
    scaler.fit(np.array(X_train))
    X_train = scaler.transform(np.array(X_train))
    # standardize X_val and X_test
    X_val = scaler.transform(np.array(X_val))
    X_test = scaler.transform(np.array(X_test))

    # Trajectory + start_frame features
    X_train = X_train[:,:201]
    X_val = X_val[:,:201]
    X_test = X_test[:,:201]

    # Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    # Dataloader 
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ### Train and Validate model
    model = MLP(output_size=len(cells_of_interest)).to(device)
    lr = 0.001
    weight_decay=0.1 # L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    epochs = 5000
    log_interval = 100 # per batch
    save_interval = 500 # per epoch
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(1, epochs + 1):
        print(f"Train epoch {epoch}, \t time {time.strftime('%H:%M:%S', time.localtime())}")
        loss_train = _utilities.train(model, train_dl, optimizer, log_interval, device)
        loss_val, accuracy_val = _utilities.test(model, val_dl, device)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_accuracy.append(accuracy_val)
        # we can save the model after each epoch
        if epoch%save_interval == 0:
            torch.save(model.state_dict(), f"./mlp/model_pt/mlp_trajectory_start_frame_{i}_fold_CV_{epoch}.pt")
    # record for this CV fold
    cv_train_loss.append(train_loss[:])
    cv_val_loss.append(val_loss[:])
    cv_val_accuracy.append(val_accuracy[:])

with open( './mlp/cv_train_loss_trajectory_start_frame.json', 'w') as f:
   json.dump(cv_train_loss, f)
with open( './mlp/cv_val_loss_trajectory_start_frame.json', 'w') as f:
   json.dump(cv_val_loss, f)
with open( './mlp/cv_val_accuracy_trajectory_start_frame.json', 'w') as f:
   json.dump(cv_val_accuracy, f)


########################################## MLP use only trajectory + start_frame + lifespan_frame ###########################################################################################
cv_train_loss = []
cv_val_loss = []
cv_val_accuracy = []
# Cross Validation
for i in range(6):
    print(f'{i}-fold')
    # train and val data
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    # test data
    X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)


    # feature normalization
    scaler = StandardScaler()
    scaler.fit(np.array(X_train))
    X_train = scaler.transform(np.array(X_train))
    # standardize X_val and X_test
    X_val = scaler.transform(np.array(X_val))
    X_test = scaler.transform(np.array(X_test))

    # Trajectory + start_frame + lifespan_frame features
    X_train = X_train[:,:202]
    X_val = X_val[:,:202]
    X_test = X_test[:,:202]

    # Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    # Dataloader 
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ### Train and Validate model
    model = MLP(output_size=len(cells_of_interest)).to(device)
    lr = 0.001
    weight_decay=0.1 # L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    epochs = 5000
    log_interval = 100 # per batch
    save_interval = 500 # per epoch
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(1, epochs + 1):
        print(f"Train epoch {epoch}, \t time {time.strftime('%H:%M:%S', time.localtime())}")
        loss_train = _utilities.train(model, train_dl, optimizer, log_interval, device)
        loss_val, accuracy_val = _utilities.test(model, val_dl, device)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_accuracy.append(accuracy_val)
        # we can save the model after each epoch
        if epoch%save_interval == 0:
            torch.save(model.state_dict(), f"./mlp/model_pt/mlp_trajectory_start_frame_lifespan_frame_{i}_fold_CV_{epoch}.pt")
    # record for this CV fold
    cv_train_loss.append(train_loss[:])
    cv_val_loss.append(val_loss[:])
    cv_val_accuracy.append(val_accuracy[:])

with open( './mlp/cv_train_loss_trajectory_start_frame_lifespan_frame.json', 'w') as f:
   json.dump(cv_train_loss, f)
with open( './mlp/cv_val_loss_trajectory_start_frame_lifespan_frame.json', 'w') as f:
   json.dump(cv_val_loss, f)
with open( './mlp/cv_val_accuracy_trajectory_start_frame_lifespan_frame.json', 'w') as f:
   json.dump(cv_val_accuracy, f)


########################################## MLP use only trajectory + start_frame + lifespan_frame + division_orientation_to_mother_cell ###########################################################################################
cv_train_loss = []
cv_val_loss = []
cv_val_accuracy = []
# Cross Validation
for i in range(6):
    print(f'{i}-fold')
    # train and val data
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    # test data
    X_test, _, y_test = _utilities.prepare_data_for_model(embryo_cells_info, embryos_for_test, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)


    # feature normalization
    scaler = StandardScaler()
    scaler.fit(np.array(X_train))
    X_train = scaler.transform(np.array(X_train))
    # standardize X_val and X_test
    X_val = scaler.transform(np.array(X_val))
    X_test = scaler.transform(np.array(X_test))

    # Trajectory + start_frame + lifespan_frame + division_orientation_to_mother_cell features
    X_train = X_train[:,:205]
    X_val = X_val[:,:205]
    X_test = X_test[:,:205]

    # Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train, dtype=np.float32)), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val, dtype=np.float32)), torch.from_numpy(np.array(y_val)).type(torch.LongTensor))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test, dtype =np.float32)), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    # Dataloader 
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ### Train and Validate model
    model = MLP(output_size=len(cells_of_interest)).to(device)
    lr = 0.001
    weight_decay=0.1 # L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    epochs = 5000
    log_interval = 100 # per batch
    save_interval = 500 # per epoch
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(1, epochs + 1):
        print(f"Train epoch {epoch}, \t time {time.strftime('%H:%M:%S', time.localtime())}")
        loss_train = _utilities.train(model, train_dl, optimizer, log_interval, device)
        loss_val, accuracy_val = _utilities.test(model, val_dl, device)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_accuracy.append(accuracy_val)
        # we can save the model after each epoch
        if epoch%save_interval == 0:
            torch.save(model.state_dict(), f"./mlp/model_pt/mlp_trajectory_and_start_frame_and_lifespan_and_division_orientation_to_mother_cell_{i}_fold_CV_{epoch}.pt")
    # record for this CV fold
    cv_train_loss.append(train_loss[:])
    cv_val_loss.append(val_loss[:])
    cv_val_accuracy.append(val_accuracy[:])

with open( './mlp/cv_train_loss_trajectory_and_start_frame_and_lifespan_and_division_orientation_to_mother_cell.json', 'w') as f:
   json.dump(cv_train_loss, f)
with open( './mlp/cv_val_loss_trajectory_and_start_frame_and_lifespan_and_division_orientation_to_mother_cell.json', 'w') as f:
   json.dump(cv_val_loss, f)
with open( './mlp/cv_val_accuracy_trajectory_and_start_frame_and_lifespan_and_division_orientation_to_mother_cell.json', 'w') as f:
   json.dump(cv_val_accuracy, f)
