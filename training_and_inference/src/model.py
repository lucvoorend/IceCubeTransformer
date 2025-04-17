import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml

import numpy as np

from src.loss import dir_3vec_loss, MSE_loss, VonMisesFisherLoss3D, opening_angle_loss, Simon_loss

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# set loss func based on the config file but not the string
if config['training_params']['loss_function'] == 'dir_3vec_loss':
    loss_func = dir_3vec_loss
elif config['training_params']['loss_function'] == 'MSE_loss':
    loss_func = MSE_loss
elif config['training_params']['loss_function'] == 'VonMisesFisherLoss3D':
    loss_func = VonMisesFisherLoss3D
elif config['training_params']['loss_function'] == 'opening_angle_loss':
    loss_func = opening_angle_loss
elif config['training_params']['loss_function'] == 'Simon_loss':
    loss_func = Simon_loss

class AttentionHead(nn.Module):
    """ 
    Single attention head class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    """
    def __init__(
            self,
            head_dim: int,
            dropout,
            ):
        super(AttentionHead, self).__init__()
        self.head_dim = head_dim
        self.query = nn.Linear(head_dim, head_dim)
        self.key = nn.Linear(head_dim, head_dim)
        self.value = nn.Linear(head_dim, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_lengths=None):
        batch_size, seq_dim, head_dim = q.shape
        

        attention_weights =  torch.matmul(q, k.transpose(-2, -1)) # compute attention weights by taking the dot product of query and key
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.head_dim).float()) # scale by dividing by sqrt(embedding_dim)

        if event_lengths is not None:
            # Step 1: Generate row and column indices for a square matrix of size seq_dim
            row_indices = torch.arange(seq_dim).view(1, -1, 1)  # Shape: (1, seq_dim, 1)
            col_indices = torch.arange(seq_dim).view(1, 1, -1)  # Shape: (1, 1, seq_dim)

            # Map row and col indices to the device of the input tensor
            row_indices = row_indices.to(q.device)
            col_indices = col_indices.to(q.device)

            # Step 2: Compare indices against event_lengths to create the mask
            event_lengths_new = event_lengths.view(-1, 1, 1).to(q.device)  # Shape: (batch_size, 1, 1)

            mask = (row_indices < event_lengths_new) & (col_indices < event_lengths_new)
            # Mask shape: (batch_size, seq_dim, seq_dim)
            #attention_weights[~mask] = float('-inf')
            attention_weights = attention_weights.masked_fill(~mask, float('-1e9'))

        attention_weights = torch.softmax(attention_weights, dim=-1) # apply softmax to attention weights
        #attention_weights[attention_weights != attention_weights] = 0 # set all nan values to 0 (result of masking entire row to -inf and then softmax)

        #attention_weights = self.dropout(attention_weights) # apply dropout to attention weights

        output = torch.matmul(attention_weights, v) # compute output by taking the dot product of attention weights and value
        # output shape: (batch_size, seq_dim, head_dim)

        return output


class MultiAttentionHead(nn.Module):
    """ 
    Multi-head attention class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    - uses multiple attention heads in parallel
    """

    def __init__(
            self, 
            embedding_dim, 
            n_heads,
            dropout,
            ):
        super(MultiAttentionHead, self).__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"

        self.embedding_dim = embedding_dim
        self.nheads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.heads = nn.ModuleList([AttentionHead(self.head_dim, dropout) for _ in range(n_heads)])

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.summarize = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, event_lengths=None):
        batch_size, seq_dim, _ = x.shape

        # Project input to queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Split the embedding into n_heads
        q = q.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)

        # Store single-head outputs in a list by looping over single-head attention layers
        head_outputs = [head(q[:, i], k[:, i], v[:, i], event_lengths=event_lengths) for i, head in enumerate(self.heads)]

        multihead_output = torch.cat(head_outputs, dim=-1) # concatenate the outputs of each head to get a single tensor
        output = self.summarize(multihead_output) # apply a linear layer to summarize the multi-head output to the original embedding dimension
        output = self.dropout(multihead_output) # apply dropout to the output

        return output

class FeedForward(nn.Module):
    """
    Feedforward network class:
    - applies a feedforward network to a sequence of embeddings
    - returns a sequence of the same length
    """

    def __init__(
            self, 
            embedding_dim,
            dropout,
            ):
        super(FeedForward, self).__init__()
        self.step = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.step(x)

class DecoderBlock(nn.Module):
    """
    Decoder block class:
    - contains a multi-head attention layer and a feedforward network
    - applies layer normalization after each sublayer
    - applies a skip connection around each sublayer
    - returns a sequence of the same length
    """

    def __init__(
            self, 
            embedding_dim,
            n_heads,
            dropout,
            ):
        super(DecoderBlock, self).__init__()
        self.multihead = MultiAttentionHead(embedding_dim, n_heads, dropout)
        self.feedforward = FeedForward(embedding_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, event_lengths=None):
        x_multi = self.multihead(x, event_lengths=event_lengths) # apply multi-head attention
        x = x + x_multi # add the multi-head output to the input (skip connection)
        x = self.norm1(x) # apply layer normalization

        x_ff = self.feedforward(x) # apply feedforward network
        x = x + x_ff # add the feedforward output to the input (skip connection)
        x = self.norm2(x) # apply layer normalization

        return x
    
class AveragePooling(nn.Module):
    """
    Average pooling class:
    - applies average pooling to a sequence of embeddings along the sequence dimension
    - returns a single embedding
    """
    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1)
    
class MaxPooling(nn.Module):
    """
    Max pooling class:
    - applies max pooling to a sequence of embeddings along the sequence dimension
    - returns a single embedding
    """
    def __init__(self):
        super(MaxPooling, self).__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)
    
class Linear_regression(nn.Module):
    """
    Linear regression class:
    - applies a linear layer to the max-pooled embedding (1 by embedding_dim)
    - returns 1 value (predicted target)
    """
    def __init__(
            self, 
            embedding_dim,
            output_dim,
            ):
        super(Linear_regression, self).__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
class regression_Transformer(nn.Module):
    """
    Regression transformer class:
    - contains an input embedding layer, position embedding layer, transformer layers, and output layers
    - applies the transformer to a sequence of embeddings
    - returns a single predicted target value
    """
    def __init__(
            self,
            embedding_dim=96,
            n_layers=6,
            n_heads=6,
            input_dim=7,
            seq_dim=256,
            dropout=0.1,
            output_dim=1,
            ):
        super(regression_Transformer, self).__init__()

        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(seq_dim, embedding_dim)
        self.layers = nn.ModuleList([DecoderBlock(embedding_dim, n_heads, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mean_pooling = AveragePooling() # average pooling layer to get a single embedding from the sequence
        self.max_pooling = MaxPooling() # max pooling layer to get a single embedding from the sequence
        self.linear_regression = Linear_regression(embedding_dim, output_dim) # linear regression layer to predict the target

    def forward(self, x, target=None, event_lengths=None):
        seq_dim_x = x.shape[1]
        device = x.device

        input_emb = self.input_embedding(x).to(device)
        #print("Input emb shape: ", input_emb.shape)
        pos_emb = self.position_embedding(torch.arange(seq_dim_x, device=device))
        pos_emb = pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1)	# Shape: (batch_size, seq_dim, emb_dim)
        
        x = input_emb + pos_emb

        #print("Final emb shape: ", x.shape)

        for layer in self.layers:
            x = layer(x, event_lengths=event_lengths)

        # Feed the output of the transformer to the pooling layer
        batch_dim, seq_dim_x, emb_dim = x.shape[0], x.shape[1], x.shape[2]

        # Aggregate x over the sequence dimension to the event length
        row_indices = torch.arange(seq_dim_x).view(1, -1, 1)  # Shape: (1, seq_dim, 1)
        row_indices = row_indices.expand(batch_dim, -1, emb_dim)

        row_indices = row_indices.to(device)

        mask = row_indices < event_lengths.view(-1, 1, 1).to(device) # Shape: (batch_size, seq_dim, emb_dim)

        # Apply mask to x
        x = x.masked_fill(mask == 0, 0)

        # Mean pooling over the sequence dimension
        x = x.sum(dim=1) / event_lengths.view(-1, 1) # Shape: (batch_size, emb_dim)

        # Feed to a linear regression layer
        y_pred = self.linear_regression(x)

        if target is None:
            loss = None
            return y_pred, loss
        
        else:
            loss = loss_func(y_pred, target)
            return y_pred, loss

#==================================================================================================
# Define the PyTorch Lightning model      
class LitModel(pl.LightningModule):
    def __init__(
            self, 
            model, 
            optimizer,
            train_dataset, 
            val_dataset,
            batch_size=16,
            ):
        super(LitModel, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        # Store the training and validation losses
        self.train_losses = []
        self.val_losses = []

        self.train_opening_angles = []
        self.val_opening_angles = []

    def forward(self, x, event_lengths=None):
        return self.model(x, event_lengths=event_lengths)

    def training_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths)
        #mean_loss = torch.mean(loss)
        self.train_losses.append(loss.item())

        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=False, logger=True, sync_dist=True)

        if batch_idx % 100 == 0:
            # Print y_pred and target for the first 5 events in the batch
            print("\n")
            print("y_pred: ", y_pred[:5])
            print("target: ", target[:5])

            pred_x = y_pred[:5, 0].detach().cpu().to(torch.float32).numpy()
            pred_y = y_pred[:5, 1].detach().cpu().to(torch.float32).numpy()
            pred_z = y_pred[:5, 2].detach().cpu().to(torch.float32).numpy()

            target_x = target[:5, 0].detach().cpu().to(torch.float32).numpy()
            target_y = target[:5, 1].detach().cpu().to(torch.float32).numpy()
            target_z = target[:5, 2].detach().cpu().to(torch.float32).numpy()

            opening_angle = np.arccos((pred_x * target_x + pred_y * target_y + pred_z * target_z) / (np.sqrt(pred_x**2 + pred_y**2 + pred_z**2) * np.sqrt(target_x**2 + target_y**2 + target_z**2))) * 180 / np.pi
            print("Opening angle (deg): ", opening_angle)

            self.train_opening_angles.append(opening_angle)

        
        self.log('train_loss', loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        # Log the median training loss at the end of each epoch
        median_train_loss = torch.tensor(self.train_losses).median().item()
        self.log('median_train_loss', median_train_loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        mean_train_opening_angle = np.mean(self.train_opening_angles)
        median_train_opening_angle = np.median(self.train_opening_angles)
        self.log('mean_train_opening_angle', mean_train_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('median_train_opening_angle', median_train_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        self.train_losses = []
        self.train_opening_angles = []
        
    def validation_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths)
        #loss = torch.mean(loss)
        self.val_losses.append(loss.item())

        if batch_idx % 100 == 0:
            # Print y_pred and target for the first 5 events in the batch
            print("\n")
            print("y_pred: ", y_pred[:5])
            print("target: ", target[:5])

        if batch_idx % 10 == 0:

            pred_x = y_pred[:5, 0].detach().cpu().to(torch.float32).numpy()
            pred_y = y_pred[:5, 1].detach().cpu().to(torch.float32).numpy()
            pred_z = y_pred[:5, 2].detach().cpu().to(torch.float32).numpy()

            target_x = target[:5, 0].detach().cpu().to(torch.float32).numpy()
            target_y = target[:5, 1].detach().cpu().to(torch.float32).numpy()
            target_z = target[:5, 2].detach().cpu().to(torch.float32).numpy()

            opening_angle = np.arccos((pred_x * target_x + pred_y * target_y + pred_z * target_z) / (np.sqrt(pred_x**2 + pred_y**2 + pred_z**2) * np.sqrt(target_x**2 + target_y**2 + target_z**2))) * 180 / np.pi
            print("Opening angle (deg): ", opening_angle)

            self.val_opening_angles.append(opening_angle)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # Log the median validation loss at the end of each epoch
        median_val_loss = torch.tensor(self.val_losses).median().item()
        self.log('median_val_loss', median_val_loss, prog_bar=True, on_epoch=True, logger=True)

        mean_val_opening_angle = np.mean(self.val_opening_angles)
        median_val_opening_angle = np.median(self.val_opening_angles)
        self.log('mean_val_opening_angle', mean_val_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('median_val_opening_angle', median_val_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        self.val_losses = []
        self.val_opening_angles = []

    def predict_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, _ = self.model(x, event_lengths=event_lengths)
        return {'y_pred': y_pred, 'target': target}

    def configure_optimizers(self):
        return self.optimizer
