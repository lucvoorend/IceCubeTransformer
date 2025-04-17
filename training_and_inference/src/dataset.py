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
    

class PMTfiedDatasetPyArrowMulti(Dataset):
    def __init__(
            self, 
            truth_paths_1,
            truth_paths_2,
            truth_paths_3,
            sample_weights = [1, 1, 1],
            selection=None,
            transform=feature_preprocessing,
    ):
        '''
        Args:
        - truth_paths: List of paths to the truth files
        - selection: List of event numbers to select from the corresponding truth files
        - transform: Function to apply to the features as preprocessing
        '''

        self.truth_paths_1 = truth_paths_1
        self.truth_paths_2 = truth_paths_2
        self.truth_paths_3 = truth_paths_3
        self.selection = selection
        self.transform = transform

        # Metadata variables
        self.event_counts = [0,0,0]
        self.cumulative_event_counts_1 = []
        self.cumulative_event_counts_2 = []
        self.cumulative_event_counts_3 = []

        self.current_file_idx_1 = None
        self.current_file_idx_2 = None
        self.current_file_idx_3 = None

        self.current_truth_1 = None
        self.current_truth_2 = None
        self.current_truth_3 = None

        self.current_feature_path_1 = None
        self.current_feature_path_2 = None
        self.current_feature_path_3 = None
     
        self.current_features_1 = None
        self.current_features_2 = None
        self.current_features_3 = None

        # Scan the truth files to get the event 
        total_events = 0
        for path in self.truth_paths_1:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            n_events = len(truth)
            self.event_counts[0] += n_events
            total_events += n_events
            self.cumulative_event_counts_1.append(total_events)


        total_events = 0
        for path in self.truth_paths_2:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            n_events = len(truth)
            self.event_counts[1] += n_events
            total_events += n_events
            self.cumulative_event_counts_2.append(total_events)


        total_events = 0
        for path in self.truth_paths_3:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            n_events = len(truth)
            self.event_counts[2] += n_events
            total_events += n_events
            self.cumulative_event_counts_3.append(total_events)

        self.sample_weights = sample_weights
        self.total_weights = sum(self.sample_weights)

        print('Total events:', self.event_counts)
        print('Cumulative event counts 1:', self.cumulative_event_counts_1)
        print('Cumulative event counts 2:', self.cumulative_event_counts_2)
        print('Cumulative event counts 3:', self.cumulative_event_counts_3)

    def __len__(self):
        # Devide the event counts per file type by the sample weights, take the minimum times the sample weights
        return min([count // weight for count, weight in zip(self.event_counts, self.sample_weights)]) * self.total_weights

    def __getitem__(self, idx):
        # Find the file index for the given event index, sampling from different truth lists
        
        set_idx = idx // self.total_weights
        mod_idx = idx % self.total_weights

        if mod_idx < self.sample_weights[0]:
            file_idx = np.searchsorted(self.cumulative_event_counts_1, self.sample_weights[0]*set_idx + mod_idx, side='right')
            local_idx = self.sample_weights[0]*set_idx + mod_idx if file_idx == 0 else self.sample_weights[0]*set_idx + mod_idx - self.cumulative_event_counts_1[file_idx - 1]
            truth_path = self.truth_paths_1[file_idx]

            if file_idx != self.current_file_idx_1:
                self.current_file_idx_1 = file_idx

                truth = pq.read_table(truth_path)
                if self.selection is not None:
                    mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                    self.current_truth_1 = truth.filter(mask)
                else:
                    self.current_truth_1 = truth

            truth = self.current_truth_1


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

            offset = int(truth.column('offset')[local_idx].as_py())
            n_doms = int(truth.column('N_doms')[local_idx].as_py())
            part_no = int(truth.column('part_no')[local_idx].as_py())
            shard_no = int(truth.column('shard_no')[local_idx].as_py())

            # Define the feature path based on the truth path
            feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))

            # x from rows (offset-n_doms) to offset
            start_row = offset - n_doms

            # Load the features and apply preprocessing
            if feature_path != self.current_feature_path_1:
                self.current_feature_path_1 = feature_path
                self.current_features_1 = pq.read_table(feature_path)

            features = self.current_features_1

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

            return Data(x=x_tensor, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy=energy, azimuth=azimuth, zenith=zenith, dir3vec=dir3vec, pid=pid)

        elif mod_idx < self.sample_weights[0] + self.sample_weights[1]:
            file_idx = np.searchsorted(self.cumulative_event_counts_2, self.sample_weights[1]*set_idx + mod_idx - self.sample_weights[0], side='right')
            local_idx = self.sample_weights[1]*set_idx + mod_idx - self.sample_weights[0] if file_idx == 0 else self.sample_weights[1]*set_idx + mod_idx - self.sample_weights[0] - self.cumulative_event_counts_2[file_idx - 1]
            truth_path = self.truth_paths_2[file_idx]

            if file_idx != self.current_file_idx_2:
                self.current_file_idx_2 = file_idx

                truth = pq.read_table(truth_path)
                if self.selection is not None:
                    mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                    self.current_truth_2 = truth.filter(mask)
                else:
                    self.current_truth_2 = truth

            truth = self.current_truth_2

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

            offset = int(truth.column('offset')[local_idx].as_py())
            n_doms = int(truth.column('N_doms')[local_idx].as_py())
            part_no = int(truth.column('part_no')[local_idx].as_py())
            shard_no = int(truth.column('shard_no')[local_idx].as_py())

            # Define the feature path based on the truth path
            feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))

            # x from rows (offset-n_doms) to offset
            start_row = offset - n_doms

            # Load the features and apply preprocessing
            if feature_path != self.current_feature_path_2:
                self.current_feature_path_2 = feature_path
                self.current_features_2 = pq.read_table(feature_path)

            features = self.current_features_2

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

            return Data(x=x_tensor, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy=energy, azimuth=azimuth, zenith=zenith, dir3vec=dir3vec, pid=pid)

        elif mod_idx < self.sample_weights[0] + self.sample_weights[1] + self.sample_weights[2]:
            file_idx = np.searchsorted(self.cumulative_event_counts_3, set_idx*self.sample_weights[2] + mod_idx - self.sample_weights[0] - self.sample_weights[1], side='right')
            local_idx = self.sample_weights[2]*set_idx + mod_idx - self.sample_weights[0] - self.sample_weights[1] if file_idx == 0 else self.sample_weights[2]*set_idx + mod_idx - self.sample_weights[0] - self.sample_weights[1] - self.cumulative_event_counts_3[file_idx - 1]
            truth_path = self.truth_paths_3[file_idx]

            if file_idx != self.current_file_idx_3:
                self.current_file_idx_3 = file_idx

                truth = pq.read_table(truth_path)
                if self.selection is not None:
                    mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                    self.current_truth_3 = truth.filter(mask)
                else:
                    self.current_truth_3 = truth

            truth = self.current_truth_3

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

            offset = int(truth.column('offset')[local_idx].as_py())
            n_doms = int(truth.column('N_doms')[local_idx].as_py())
            part_no = int(truth.column('part_no')[local_idx].as_py())
            shard_no = int(truth.column('shard_no')[local_idx].as_py())

            # Define the feature path based on the truth path
            feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))

            # x from rows (offset-n_doms) to offset
            start_row = offset - n_doms

            # Load the features and apply preprocessing
            if feature_path != self.current_feature_path_3:
                self.current_feature_path_3 = feature_path
                self.current_features_3 = pq.read_table(feature_path)

            features = self.current_features_3

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

            return Data(x=x_tensor, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy=energy, azimuth=azimuth, zenith=zenith, dir3vec=dir3vec, pid=pid)
