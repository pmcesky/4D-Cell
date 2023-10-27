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