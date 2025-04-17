import torch
from torch.utils.data import DataLoader
import yaml

import numpy as np
import pandas as pd

from src.dataset import PMTfiedDatasetPyArrow
from src.dataset import PMTfiedDatasetPyArrowMulti

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

reconstruction_target = config['input_data']['reconstruction_target']

def pad_or_truncate(event, max_seq_length=config['input_data']['seq_dim'], total_charge_index=int(16)):
    """
    Pad or truncate an event to the given max sequence length, and create an attention mask.
    
    Args:
    - event: Tensor of shape (seq_length, feature_dim) where seq_length can vary.
    - max_seq_length: Maximum sequence length to pad/truncate to.
    
    Returns:
    - Padded or truncated event of shape (max_seq_length, feature_dim).
    - Attention mask of shape (max_seq_length) where 1 indicates a valid token and 0 indicates padding.
    """
    seq_length = event.size(0)
    
    # Truncate if the sequence is too long
    if seq_length > max_seq_length:
        # sort the event by total charge
        event = event[event[:, total_charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    

    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    
    # No need to pad or truncate if it's already the correct length
    return event, seq_length

def custom_collate_fn(batch, max_seq_length=config['input_data']['seq_dim']):
    """
    Custom collate function to pad or truncate each event in the batch.
    
    Args:
    - batch: List of (event, label) tuples where event has a variable length [seq_length, 7].
    - max_seq_length: The fixed length to pad/truncate each event to (default is 512).
    
    Returns:
    - A batch of padded/truncated events with shape [batch_size, max_seq_length, 7].
    - Corresponding labels.
    """
    # Separate events and labels
    events = [item.x for item in batch]  # Each event has shape [seq_length, 7]
    
    padded_events, event_lengths = zip(*[pad_or_truncate(event, max_seq_length) for event in events])

    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)

    # Extract labels and convert to tensors
    label_name = reconstruction_target

    # Extract labels and convert to tensors (3D vectors)
    vectors = [item[label_name] for item in batch]

    # Stack labels in case of multi-dimensional output
    labels = torch.stack(vectors)

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    
    return batch_events, labels, event_lengths

def pad_or_truncate_pulse(event, max_seq_length=config['input_data']['seq_dim'], charge_index=int(0)):
    """
    Pad or truncate an event to the given max sequence length, and create an attention mask.
    
    Args:
    - event: Tensor of shape (seq_length, feature_dim) where seq_length can vary.
    - max_seq_length: Maximum sequence length to pad/truncate to.
    
    Returns:
    - Padded or truncated event of shape (max_seq_length, feature_dim).
    - Attention mask of shape (max_seq_length) where 1 indicates a valid token and 0 indicates padding.
    """
    seq_length = event.size(0)
    
    # Truncate if the sequence is too long
    if seq_length > max_seq_length:
        # sort the event by total charge
        event = event[event[:, charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    

    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    
    # No need to pad or truncate if it's already the correct length
    return event, seq_length

def custom_collate_fn_pulse(batch, max_seq_length=config['input_data']['seq_dim']):
    """
    Custom collate function to pad or truncate each event in the batch.
    
    Args:
    - batch: List of (event, label) tuples where event has a variable length [seq_length, 7].
    - max_seq_length: The fixed length to pad/truncate each event to (default is 512).
    
    Returns:
    - A batch of padded/truncated events with shape [batch_size, max_seq_length, 7].
    - Corresponding labels.
    """
    # Separate events and labels
    events = [item.x for item in batch]  # Each event has shape [seq_length, 7]
    
    padded_events, event_lengths = zip(*[pad_or_truncate_pulse(event, max_seq_length) for event in events])

    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)

    if reconstruction_target == 'dir3vec': # exception for dir3vec pulse, since graphnet does not have dir3vec in dataset class
        zenith = torch.tensor([item['zenith'] for item in batch])
        azimuth = torch.tensor([item['azimuth'] for item in batch])

        # Calculate a 3D unit-vector from the zenith and azimuth angles
        x_dir = torch.sin(zenith) * torch.cos(azimuth)
        y_dir = torch.sin(zenith) * torch.sin(azimuth)
        z_dir = torch.cos(zenith)

        # Stack to dir3vec tensor
        labels = torch.stack([x_dir, y_dir, z_dir], dim=-1)

    else:
        # Extract labels and convert to tensors
        label_name = reconstruction_target

        # Extract labels and convert to tensors (3D vectors)
        vectors = [item[label_name] for item in batch]

        # Stack labels in case of multi-dimensional output
        labels = torch.stack(vectors)

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    
    return batch_events, labels, event_lengths

def make_dataloader_PMT(
        root_dir,
        dataset_id,
        training_parts,
        validation_parts,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
):
    """
    Create a DataLoader for the PMTfied dataset. Takes data from a single dataset ID.
    
    Args:
    - root_dir: Root directory of the dataset.
    - dataset_id: ID of the dataset.
    - training_parts: List of training parts.
    - validation_parts: List of validation parts.
    
    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """

    # unpack the training and validation parts such that you can either input a list of lists or a single list
    if isinstance(training_parts[0], list):
        training_parts = np.array(training_parts[0])
    else:
        training_parts = np.array(training_parts)
    if isinstance(validation_parts[0], list):
        validation_parts = np.array(validation_parts[0])
    else:
        validation_parts = np.array(validation_parts)

    train_paths = []
    val_paths = []
    for part in training_parts:
        train_paths.append(f"{root_dir}/{dataset_id}/truth_{part}.parquet")
    for part in validation_parts:
        val_paths.append(f"{root_dir}/{dataset_id}/truth_{part}.parquet")
        


    train_set = PMTfiedDatasetPyArrow(
    truth_paths=train_paths,
    )

    val_set = PMTfiedDatasetPyArrow(
    truth_paths=val_paths,
    )

    train_loader = DataLoader(
        dataset=train_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader

def make_dataloader_PMT_multi(
        root_dir,
        dataset_ids,
        training_parts,
        validation_parts,
        sample_weights=[1, 1, 1], # for the three datasets
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
):
    """
    Create a DataLoader for the PMTfied dataset. Takes data from multiple dataset IDs.
    
    Args:
    - root_dir: Root directory of the dataset.
    - dataset_ids: List of dataset IDs.
    - training_parts: List of training parts.
    - validation_parts: List of validation parts.
    
    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """
    train_paths_1 = []
    val_paths_1 = []

    train_paths_2 = []
    val_paths_2 = []

    train_paths_3 = []
    val_paths_3 = []

    # It isn't pretty, but it works
    for part in training_parts[0]:
        train_paths_1.append(f"{root_dir}/{dataset_ids[0]}/truth_{part}.parquet")
    for part in validation_parts[0]:
        val_paths_1.append(f"{root_dir}/{dataset_ids[0]}/truth_{part}.parquet")

    for part in training_parts[1]:
        train_paths_2.append(f"{root_dir}/{dataset_ids[1]}/truth_{part}.parquet")
    for part in validation_parts[1]:
        val_paths_2.append(f"{root_dir}/{dataset_ids[1]}/truth_{part}.parquet")

    for part in training_parts[2]:
        train_paths_3.append(f"{root_dir}/{dataset_ids[2]}/truth_{part}.parquet")
    for part in validation_parts[2]:
        val_paths_3.append(f"{root_dir}/{dataset_ids[2]}/truth_{part}.parquet")

    train_set = PMTfiedDatasetPyArrowMulti(
    truth_paths_1=train_paths_1,
    truth_paths_2=train_paths_2,
    truth_paths_3=train_paths_3,
    sample_weights=sample_weights,
    )

    val_set = PMTfiedDatasetPyArrowMulti(
    truth_paths_1=val_paths_1,
    truth_paths_2=val_paths_2,
    truth_paths_3=val_paths_3,
    sample_weights=sample_weights,
    )

    train_loader = DataLoader(
        dataset=train_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader, val_loader

def make_dataloader_PMT_inference(
        root_dir,
        dataset_id,
        inference_parts,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
):
    """
    Create a DataLoader for the PMTfied dataset for inference. Takes data from a single dataset ID.
    
    Args:
    - root_dir: Root directory of the dataset.
    - dataset_id: ID of the dataset.
    - inference_parts: List of dataset parts for inference.

    Returns:
    - inference_loader: DataLoader for inferenceg data.
    """

    # unpack the training and validation parts such that you can either input a list of lists or a single list
    if isinstance(inference_parts[0], list):
        inference_parts = np.array(inference_parts[0])
    else:
        inference_parts = np.array(inference_parts)


    inference_paths = []
    for part in inference_parts:
        inference_paths.append(f"{root_dir}/{dataset_id[0]}/truth_{part}.parquet")

    print(f"Loading inference data from {inference_paths}")
    event_no_array = np.array([])
    for path in inference_paths:
        pd_truth = pd.read_parquet(path)
        event_no = pd_truth['event_no'].values
        # append to the event_no array
        event_no_array = np.append(event_no_array, event_no)

    # take first 500_000 events
    if len(event_no_array) > 500_000:
        print(f"WARNING: Taking first 500_000 events from {len(event_no_array)} events because of memory constraints")
        event_no_array = event_no_array[:500_000]
        
    inference_set = PMTfiedDatasetPyArrow(
    truth_paths=inference_paths,
    selection=event_no_array,
    )

    inference_loader = DataLoader(
        dataset=inference_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return inference_loader, event_no_array
