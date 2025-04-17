print("Start importing")
import torch
import pandas as pd
import yaml

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning import Trainer

from src.model import regression_Transformer, LitModel 

from src.utils import assert_config_inference 
from src.dataloader import make_dataloader_PMT_inference 

print("Importing done")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Validate the config file
assert_config_inference(config)

#==================================================================================================

device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')

#==================================================================================================

inference_dataloader, event_no = make_dataloader_PMT_inference(
        root_dir=config['inference_params']['inference_root_dir'],
        dataset_id = config['inference_params']['inference_dataset_id'],
        inference_parts = config['inference_params']['inference_parts'],
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
    )

#==================================================================================================
# Define the model

model = regression_Transformer(
    embedding_dim = config['model_params']['embedding_dim'], 
    n_layers = config['model_params']['n_layers'], 
    n_heads = config['model_params']['n_heads'], 
    input_dim = config['model_params']['feature_dim'], 
    seq_dim = config['input_data']['seq_dim'], 
    dropout = config['model_params']['dropout'], 
    output_dim = config['model_params']['output_dim'], 
).to(device)

#==================================================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])

lit_model = LitModel(
    model,
    optimizer,
    None,
    inference_dataloader,
    batch_size=config['training_params']['batch_size'],
)
callbacks = TQDMProgressBar()

trainer = Trainer(
    accelerator= 'gpu' if 'cuda' in config['training_params']['device'] else 'cpu', 
    devices = [int(config['training_params']['device'].split(':')[-1])] if 'cuda' in config['training_params']['device'] else 1, 
    max_epochs=config['training_params']['n_epochs'], 
    log_every_n_steps=config['log_every_n_steps'], 
    logger=None,
    callbacks=callbacks,
)

#==================================================================================================
ckpt_path = config['checkpoint_path']

#==================================================================================================
# Start the inference

print('Start predicting')
predictions = trainer.predict(
    model = lit_model,
    dataloaders = inference_dataloader,
    ckpt_path = ckpt_path,
    )

print('Predictions done')

print('Start storing the predictions')
# Storing the predictions in a pandas dataframe
pred_x = []
pred_y = []
pred_z = []

target_x = []
target_y = []
target_z = []

# Loop over the predictions
for i in range(len(predictions)):
    y_pred = predictions[i]['y_pred']
    target = predictions[i]['target']
    if i == 0:
        print('y_pred', y_pred)
        print('target', target)

    # Append (batch_size, 1) to the list
    pred_x.append(y_pred[:, 0])
    pred_y.append(y_pred[:, 1])
    pred_z.append(y_pred[:, 2])

    target_x.append(target[:, 0])
    target_y.append(target[:, 1])
    target_z.append(target[:, 2])

# Concatenate the list of tensors to a single tensor
pred_x = torch.cat(pred_x, dim=0)
pred_y = torch.cat(pred_y, dim=0)
pred_z = torch.cat(pred_z, dim=0)

target_x = torch.cat(target_x, dim=0)
target_y = torch.cat(target_y, dim=0)
target_z = torch.cat(target_z, dim=0)

print('Pred_x shape:', pred_x.shape)

df = pd.DataFrame({"event_no": event_no, "x_pred": pred_x, "y_pred": pred_y, "z_pred": pred_z, "x_truth": target_x, "y_truth": target_y, "z_truth": target_z})

destination = config['inference_params']['inference_output_path'] + config['run_name'] + '_' +  str(config['inference_params']['inference_dataset_id']) + '_' + str(config['inference_params']['inference_parts'])+ '_prediction.csv'
df.to_csv(destination, index=False)
print('Predictions stored')
