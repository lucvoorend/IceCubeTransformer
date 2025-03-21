'''
Author: Luc Voorend
Date: 2025-03-18
Description: This script defines a PyTorch dataset class for processing IceCube PMT data using PyArrow.
It loads event truth data from Parquet files, applies feature preprocessing, and structures the data for 
use in graph-based deep learning models with PyTorch Geometric. 

The dataset efficiently handles large-scale data by dynamically loading features and caching them as needed.
Normalization is applied to key input features to ensure stable training. 

Usage: Provide a list of truth file paths as input. The dataset will automatically handle filtering, 
feature transformation, and retrieval of event details. Ensure that PyArrow, NumPy, and PyTorch Geometric 
are installed before use.
'''

import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def feature_preprocessing(col_name, value) -> np.ndarray:
    """
    Preprocess the input features when creating the dataset.
    
    Args:
    - col_name: The name of the column to preprocess.
    - value: The value array of the column to preprocess [numpy array].

    Returns:
    - The preprocessed value array [numpy array].
    """
  
    if col_name in ['dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel']:
        value = value / 500
    elif col_name in ['rde']:
        value = (value - 1.25) / 0.25
    elif col_name in ['pmt_area']:
        value = value / 0.05
    elif col_name in ['q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal']:
        mask = value > 0
        value[mask] = np.log10(value[mask])
    elif col_name in ['t1', 't2', 't3','t4', 't5']:
        mask = value > 0
        value[mask] = (value[mask] - 1.0e04) / 3.0e04
    elif col_name in ['T10', 'T50', 'sigmaT']:
        mask = value > 0
        value[mask] = value[mask] / 1.0e04

    return value


class PMTfiedDatasetPyArrow(Dataset):
    def __init__(
            self, 
            truth_paths,
            selection=None,
            transform=feature_preprocessing,
    ):
        '''
        Args:
        - truth_paths: List of paths to the truth files
        - selection: List of event numbers to select from the corresponding truth files
        - transform: Function to apply to the features as preprocessing
        '''

        self.truth_paths = truth_paths
        self.selection = selection
        self.transform = transform

        # Metadata variables
        self.event_counts = []
        self.cumulative_event_counts = []
        self.current_file_idx = None
        self.current_truth = None
        self.current_feature_path = None
        self.current_features = None
        total_events = 0

        # Scan the truth files to get the event counts
        for path in self.truth_paths:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            n_events = len(truth)
            self.event_counts.append(n_events)
            total_events += n_events
            self.cumulative_event_counts.append(total_events)

        self.total_events = total_events

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        # Find the corresponding file index
        file_idx = np.searchsorted(self.cumulative_event_counts, idx, side='right')
        
        # Define the truth paths
        truth_path = self.truth_paths[file_idx]

        # Define the local event index
        local_idx = idx if file_idx == 0 else idx - self.cumulative_event_counts[file_idx - 1]

        # Load the truth and apply selection
        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            
            truth = pq.read_table(truth_path)
            #print("Loaded truth table")
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                self.current_truth = truth.filter(mask)
            else:
                self.current_truth = truth
            
        truth = self.current_truth

        # Get the event details
        event_no = torch.tensor(int(truth.column('event_no')[local_idx].as_py()), dtype=torch.long)
        energy = torch.tensor(truth.column('energy')[local_idx].as_py(), dtype=torch.float32)
        azimuth = torch.tensor(truth.column('azimuth')[local_idx].as_py(), dtype=torch.float32)
        zenith = torch.tensor(truth.column('zenith')[local_idx].as_py(), dtype=torch.float32)
        pid = torch.tensor(truth.column('pid')[local_idx].as_py(), dtype=torch.float32)

        # Calculate a 3D unit-vector from the zenith and azimuth angles
        x_dir = torch.sin(zenith) * torch.cos(azimuth)
        y_dir = torch.sin(zenith) * torch.sin(azimuth)
        z_dir = torch.cos(zenith)

        # Stack to dir3vec tensor
        dir3vec = torch.stack([x_dir, y_dir, z_dir], dim=-1)

        x_dir_lepton = torch.tensor(truth.column('dir_x_GNHighestEDaughter')[local_idx].as_py(), dtype=torch.float32)
        y_dir_lepton = torch.tensor(truth.column('dir_y_GNHighestEDaughter')[local_idx].as_py(), dtype=torch.float32)
        z_dir_lepton = torch.tensor(truth.column('dir_z_GNHighestEDaughter')[local_idx].as_py(), dtype=torch.float32)

        dir3vec_lepton = torch.stack([x_dir_lepton, y_dir_lepton, z_dir_lepton], dim=-1)

        offset = int(truth.column('offset')[local_idx].as_py())
        n_doms = int(truth.column('N_doms')[local_idx].as_py())
        part_no = int(truth.column('part_no')[local_idx].as_py())
        shard_no = int(truth.column('shard_no')[local_idx].as_py())

        # Define the feature path based on the truth path
        feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))

        # x from rows (offset-n_doms) to offset
        start_row = offset - n_doms

        # Load the features and apply preprocessing
        if feature_path != self.current_feature_path:
            self.current_feature_path = feature_path
            self.current_features = pq.read_table(feature_path)

        features = self.current_features

        x = features.slice(start_row, n_doms)
        # drop the first two columns (event_no and original_event_no)
        x = x.drop_columns(['event_no', 'original_event_no'])
        num_columns = x.num_columns

        x_tensor = torch.full((n_doms, num_columns), fill_value=torch.nan, dtype=torch.float32)

        for i, col_name in enumerate(x.column_names):
            value = x.column(i).to_numpy()
            value = value.copy()
            value = self.transform(col_name, value)
            # convert to torch tensor
            value_tensor = torch.from_numpy(value)
            x_tensor[:, i] = value_tensor

        return Data(x=x_tensor, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy=energy, azimuth_neutrino=azimuth, zenith_neutrino=zenith, dir3vec=dir3vec, dir3vec_lepton=dir3vec_lepton, pid=pid)
