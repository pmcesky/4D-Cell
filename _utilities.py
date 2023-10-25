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