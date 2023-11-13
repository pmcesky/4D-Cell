import numpy as np 
import pandas as pd 
import networkx as nx 
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from pathlib import Path
from tqdm import tqdm
import time
import json
import pickle
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from sklearn.preprocessing import StandardScaler



def prepare_data_for_model(embryo_cells_info, embryo_samples, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = False):
    """
    Prepare the data for model training and testing. The function will extract data from embryo_cells_info based on embryo_samples.

    Parameters
    ----------
    embryo_cells_info : a dict of dict
        a dict of dict. Outer dict key is embryo_name, inner dict key is cell name, values are cell_features in dict.
    embryo_samples : a list of embryo names.
        a list of inquired embryo names, this will be used to extract the data from embryo_cells_info corresponding to inquired embryos.
    use_frame : bool, optional
        either use frame or use time version in trajectory, start_time and lifespan, by default True. 
        If use_frame, then related features will be trajectory_processed, start_frame, and lifespan_frame;
        If not use_frame, then related features will be trajectory_processed_txyz, start_time, and lifespan_time; 
    lifespan_frames_longest : int, optional
        will append the trajectory at the end if the trajectory is shorter than this number, by default 50.
    preserve_time_dimension : bool, optional
        either preserve time/frame dimension in trajectory, by default True. 
        If true, the trajectory (frame, x, y, z) or (t, x, y, z) will be preserved, otherwise only (x, y, z) will show up.
    flatten: bool, optional
        either flatten cell_features, by default False
        if False, cell_features will be a list of list, each element in inner list will be numpy array or scaler. 
        If True, cell features will be a list of 1D array, all elements in original list will be flatten and append in order.

    Return
    ----------
    cell_features: a list of list
        the inner list is a list of cell features of one cell, ordered in [trajectory, start_time, lifespan, division_orientation_to_mother_cells, division_orientation_to_daughter_cells].
    cell_names: a list of cell names.
        same order as cell_features.
    cell_names_to_integers: a list of integers representing cell names.
        same order as cell names appear in embryo_cells_info, starting from 0. This makes it easy for machine learning tasks.
    """ 
    cell_names = []
    cell_names_to_integers = []
    cell_features = []
    assert set(embryo_cells_info.keys()) >= set(embryo_samples)
    for embryo_name in embryo_samples:
        for idx, cell in enumerate(embryo_cells_info[embryo_name]):
            cell_names.append(cell)
            cell_names_to_integers.append(idx)
            current_cell_info =  embryo_cells_info[embryo_name][cell]
            if use_frame:
                features = [current_cell_info['trajectory_processed'].to_numpy(),\
                    current_cell_info['start_frame'],current_cell_info['lifespan_frame'],\
                    current_cell_info['division_orientation_to_mother_cell'],current_cell_info['division_orientation_of_daughter_cells']]
            else:
                features = [current_cell_info['trajectory_processed_txyz'].to_numpy(),\
                    current_cell_info['start_time'],current_cell_info['lifespan_time'],\
                    current_cell_info['division_orientation_to_mother_cell'],current_cell_info['division_orientation_of_daughter_cells']]
            # append trajectory lifespan to lifespan_frame_longest
            if current_cell_info['trajectory_processed'].shape[0]<lifespan_frame_longest:
                # append -10000 at the end
                padded = -10000*np.ones((lifespan_frame_longest,4)).astype(float)
                padded[:features[0].shape[0],:] = features[0] 
                features[0] = padded.copy()
            if not preserve_time_dimension: # discard frame/time dimension
                features[0] = features[0][:,1:] # (x,y,z)
            if flatten:
                features_flattened = features[0].flatten()
                features_flattened = np.append(features_flattened, [features[1], features[2]])
                features_flattened = np.append(features_flattened, features[3].flatten())
                features_flattened = np.append(features_flattened, features[4].flatten())
                features_flattened = features_flattened.astype(float)
                features = features_flattened.copy()
            cell_features.append(features.copy())
    return cell_features, cell_names, cell_names_to_integers


def train(model, train_loader, optimizer, device, trajectory_feature_size = 150, loss_fn = F.cross_entropy, l2_lambda_output_layer = None, clip_grad_norm=None):
    model.train()
    total_acc, total_loss = 0, 0
    for data, target in train_loader:
        # move to device, usually device is cuda
        data, target = data.to(device), target.to(device)
        # partition data into trajectory and other features
        x_traj = data[:,:trajectory_feature_size] # trajectory features (50x3), 50 frames of (x,y,z)
        x_extra = data[:,trajectory_feature_size:] # extra features, like start_frame, lifespan, division orientations
        if x_extra.numel() == 0:
            x_extra = None
        optimizer.zero_grad()
        output = model(x_traj, x_extra)
        loss = loss_fn(output, target)
        if l2_lambda_output_layer: # l2 regularization on the final output layer
            l2_reg = sum(p.pow(2).sum() for p in model.fc_out.parameters())
            loss = loss + l2_lambda_output_layer*l2_reg
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        # log for train_loss of the epoch
        total_loss += loss.item()*len(data)
        # log train_acc
        pred = torch.argmax(output, axis=1)
        total_acc += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(train_loader.dataset) # average loss on whole train dataset
    total_acc /= len(train_loader.dataset)
    return total_acc, total_loss


def test(model, test_loader, device, loss_fn = F.cross_entropy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # deactivate autograd
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_traj = data[:,:200] # trajectory features
            x_extra = data[:,200:] # extra features, like start_frame, lifespan, division orientations
            if x_extra.numel() == 0:
                x_extra = None
            output = model(x_traj, x_extra)
            # sum up batch loss
            test_loss += loss_fn(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # count the correct ones
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset) # the average loss on whole test_set
    accuracy = correct/len(test_loader.dataset) # this is the accuracy on whole test_set
    print(f'Average loss: {test_loss :.4f}, Accuracy: {accuracy}')
    return test_loss, accuracy


def train_mlp(model, train_loader, optimizer, device, loss_fn = F.cross_entropy):
    model.train()
    total_acc, total_loss = 0, 0
    for data, target in train_loader:
        # move to device, usually device is cuda
        data, target = data.to(device), target.to(device)
        # partition data into trajectory and other features
        x_traj = data[:,:200] # trajectory features (50x4), 50 frames of (t,x,y,z)
        x_extra = data[:,200:] # extra features, like start_frame, lifespan, division orientations
        if x_extra.numel() == 0:
            x_extra = None
        optimizer.zero_grad()
        output = model(x_traj, x_extra)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # log for train_loss of the epoch
        total_loss += loss.item()*len(data)
        # log train_acc
        pred = torch.argmax(output, axis=1)
        total_acc += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(train_loader.dataset) # average loss on whole train dataset
    total_acc /= len(train_loader.dataset)
    return total_acc, total_loss

def evaluate_mlp(model, dataloader, optimizer, device, loss_fn = F.cross_entropy):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            # move to device, usually device is cuda
            data, target = data.to(device), target.to(device)
            # partition data into trajectory and other features
            x_traj = data[:,:200] # trajectory features (50x4), 50 frames of (t,x,y,z)
            x_extra = data[:,200:] # extra features, like start_frame, lifespan, division orientations
            if x_extra.numel() == 0:
                x_extra = None
            output = model(x_traj, x_extra)
            loss = loss_fn(output, target)
            pred = torch.argmax(output, axis=1)
            total_acc += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()*target.size(0)
    total_loss /= len(dataloader.dataset) # average loss on whole train dataset
    total_acc /= len(dataloader.dataset)
    return total_acc, total_loss


def train_rnn(model, train_loader, optimizer, device, loss_fn = F.cross_entropy, gradient_clip=False):
    model.train()
    total_acc, total_loss = 0, 0
    for data, target in train_loader:
        # move to device, usually device is cuda
        data, target = data.to(device), target.to(device)
        # partition data into trajectory and other features
        x_traj = data[:,:150] # trajectory features (50x3), 50 frames of (x,y,z)
        x_extra = data[:,150:] # extra features, like start_frame, lifespan, division orientations
        if x_extra.numel() == 0:
            x_extra = None
        optimizer.zero_grad()
        output = model(x_traj, x_extra)
        loss = loss_fn(output, target)
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        # log for train_loss of the epoch
        total_loss += loss.item()*len(data)
        # log train_acc
        pred = torch.argmax(output, axis=1)
        total_acc += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(train_loader.dataset) # average loss on whole train dataset
    total_acc /= len(train_loader.dataset)
    return total_acc, total_loss


def evaluate_rnn(model, dataloader, optimizer, device, loss_fn = F.cross_entropy):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            # move to device, usually device is cuda
            data, target = data.to(device), target.to(device)
            # partition data into trajectory and other features
            x_traj = data[:,:150] # trajectory features (50x3), 50 frames of (x,y,z)
            x_extra = data[:,150:] # extra features, like start_frame, lifespan, division orientations
            if x_extra.numel() == 0:
                x_extra = None
            output = model(x_traj, x_extra)
            loss = loss_fn(output, target)
            pred = torch.argmax(output, axis=1)
            total_acc += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()*target.size(0)
    total_loss /= len(dataloader.dataset) # average loss on whole train dataset
    total_acc /= len(dataloader.dataset)
    return total_acc, total_loss


def plot_cell_trajectory(cells_info, cell, figsize = (10,10), azim=225, save_path = None):
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection='3d')
    # ax.grid()
    for embryo_name in cells_info[cell]['trajectory_processed']:
        xyz = cells_info[cell]['trajectory_processed'][embryo_name][['x','y','z']].to_numpy()
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2] 
        ax.plot3D(x, y, z)
    ax.set_title(f'{cell}')
    # Set axes label
    ax.set_xlabel(r'x / A-P $(\mu m)$', labelpad=20)
    ax.set_ylabel(r'y / L-R $(\mu m)$', labelpad=20)
    ax.set_zlabel(r'z / D-V $(\mu m)$', labelpad=20)
    ax.view_init(azim = azim)
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_cell_division_orientation_to_mother_cell(cells_info, cell, figsize = (10,10), azim=225, save_path = None):
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection='3d')
    for embryo_name in cells_info[cell]['division_orientation_to_mother_cell']:
        division_orientation = cells_info[cell]['division_orientation_to_mother_cell'][embryo_name]
        division_orientation = np.vstack([np.zeros(3), division_orientation])
        x = division_orientation[:,0]
        y = division_orientation[:,1]
        z = division_orientation[:,2]
        ax.plot3D(x, y, z)
    ax.set_title(f'{cell}')
    # Set axes label
    ax.set_xlabel(r'x / A-P $(\mu m)$', labelpad=20)
    ax.set_ylabel(r'y / L-R $(\mu m)$', labelpad=20)
    ax.set_zlabel(r'z / D-V $(\mu m)$', labelpad=20)
    ax.view_init(azim = azim)
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_cell_division_orientation_of_daughter_cells(cells_info, cell, figsize = (10,10), azim=225, save_path = None):
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection='3d')
    for embryo_name in cells_info[cell]['division_orientation_of_daughter_cells']:
        division_orientation = cells_info[cell]['division_orientation_of_daughter_cells'][embryo_name]
        division_orientation = np.vstack([division_orientation[0,:],np.zeros(3),division_orientation[1,:]]) # add origin --- mother cell last frame position
        ax.plot3D(division_orientation[:,0],division_orientation[:,1],division_orientation[:,2])
    ax.set_title(f'{cell}')
    # Set axes label
    ax.set_xlabel(r'x / A-P $(\mu m)$', labelpad=20)
    ax.set_ylabel(r'y / L-R $(\mu m)$', labelpad=20)
    ax.set_zlabel(r'z / D-V $(\mu m)$', labelpad=20)
    ax.view_init(azim = azim)
    if save_path:
        fig.savefig(save_path)
    plt.show()