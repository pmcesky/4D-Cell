import numpy as np
import pandas as pd
import networkx as nx 
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import _utilities


with open( './cells_of_interest.json', 'r') as f:
   cells_of_interest = json.load(f)

with open('./cells_info.pickle', 'rb') as f:
    cells_info = pickle.load(f)

with open('./embryo_cells_info.pickle', 'rb') as f:
    embryo_cells_info = pickle.load(f)

embryos_for_test = ['WT-EMB05','WT-EMB12','WT-EMB19','WT-EMB26']
embryos_for_cross_validation = [embryo_name for embryo_name in embryo_cells_info if embryo_name not in embryos_for_test]


# changing n_estimators in the forest
rf_cv_score = []
for i in tqdm(range(6)): # 6-fold cross-validation
    # train and val dataset
    embryos_val = embryos_for_cross_validation[4*i:4*i+4]
    embryos_train = [embryo_name for embryo_name in embryos_for_cross_validation if embryo_name not in embryos_val]
    X_train, _, y_train = _utilities.prepare_data_for_model(embryo_cells_info, embryos_train, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_val, _, y_val = _utilities.prepare_data_for_model(embryo_cells_info, embryos_val, use_frame = True, lifespan_frame_longest = 50, preserve_time_dimension = True, flatten = True)
    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    # X_val, y_val = shuffle(X_val, y_val, random_state=1)
    scores = []
    for n_trees in range(10, 201):
        forest = RandomForestClassifier(n_estimators=n_trees, random_state=1, n_jobs=100)
        forest.fit(X_train, y_train)
        scores.append(forest.score(X_val, y_val))
    rf_cv_score.append(scores.copy())


with open( './rf_cv_score_all_features.json', 'w') as f:
   json.dump(rf_cv_score, f)