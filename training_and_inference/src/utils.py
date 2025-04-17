import re # Import regex for device check
import os # Import os for file path checks

def assert_config_train(config):
    """
    Validates the configuration dictionary for the training script.

    Args:
        config (dict): The loaded configuration dictionary.

    Raises:
        AssertionError: If any configuration check fails.
    """
    print("Validating training configuration...")

    # --- Top Level Keys ---
    required_top_keys = ['run_name', 'checkpoint_path', 'input_data', 'model_params', 'training_params']
    for key in required_top_keys:
        assert key in config, f"Missing required top-level key: '{key}'"

    # --- General Run Settings ---
    assert isinstance(config.get('run_name'), str), "'run_name' must be a string."
    # checkpoint_path can be None or str
    assert config.get('checkpoint_path') is None or isinstance(config.get('checkpoint_path'), str), \
        "'checkpoint_path' must be a string or None."

    # --- Input Data Parameters ---
    assert 'input_data' in config and isinstance(config['input_data'], dict), "'input_data' section is missing or not a dictionary."
    input_data = config['input_data']
    required_input_keys = ['root_dir', 'multi_dataset', 'dataset_ids', 'training_parts', 'validation_parts',
                           'reconstruction_target', 'seq_dim', 'num_workers']
    for key in required_input_keys:
        assert key in input_data, f"Missing key in 'input_data': '{key}'"

    assert isinstance(input_data.get('root_dir'), str), "'input_data.root_dir' must be a string."
    assert isinstance(input_data.get('multi_dataset'), bool), "'input_data.multi_dataset' must be a boolean."
    assert isinstance(input_data.get('dataset_ids'), list), "'input_data.dataset_ids' must be a list."
    assert all(isinstance(x, int) for x in input_data['dataset_ids']), "'input_data.dataset_ids' must contain only integers."
    assert isinstance(input_data.get('training_parts'), list) and all(isinstance(sublist, list) for sublist in input_data['training_parts']), \
        "'input_data.training_parts' must be a list of lists."
    assert all(all(isinstance(x, int) for x in sublist) for sublist in input_data['training_parts']), \
        "'input_data.training_parts' sublists must contain only integers."
    assert isinstance(input_data.get('validation_parts'), list) and all(isinstance(sublist, list) for sublist in input_data['validation_parts']), \
        "'input_data.validation_parts' must be a list of lists."
    assert all(all(isinstance(x, int) for x in sublist) for sublist in input_data['validation_parts']), \
        "'input_data.validation_parts' sublists must contain only integers."
    assert isinstance(input_data.get('reconstruction_target'), str), "'input_data.reconstruction_target' must be a string."
    assert isinstance(input_data.get('seq_dim'), int) and input_data['seq_dim'] > 0, \
        "'input_data.seq_dim' must be a positive integer."
    assert isinstance(input_data.get('num_workers'), int) and input_data['num_workers'] >= 0, \
        "'input_data.num_workers' must be a non-negative integer."
    if input_data['multi_dataset'] is False:
         assert len(input_data['dataset_ids']) == 1, "If 'multi_dataset' is false, 'dataset_ids' must contain exactly one ID."
         assert len(input_data['training_parts']) == 1, "If 'multi_dataset' is false, 'training_parts' must contain exactly one list."
         assert len(input_data['validation_parts']) == 1, "If 'multi_dataset' is false, 'validation_parts' must contain exactly one list."
    else:
         assert len(input_data['dataset_ids']) == len(input_data['training_parts']), \
             "Length of 'dataset_ids' must match length of 'training_parts' when 'multi_dataset' is true."
         assert len(input_data['dataset_ids']) == len(input_data['validation_parts']), \
             "Length of 'dataset_ids' must match length of 'validation_parts' when 'multi_dataset' is true."


    # --- Model Parameters ---
    assert 'model_params' in config and isinstance(config['model_params'], dict), "'model_params' section is missing or not a dictionary."
    model_params = config['model_params']
    required_model_keys = ['feature_dim', 'embedding_dim', 'output_dim', 'n_layers', 'n_heads', 'dropout']
    for key in required_model_keys:
        assert key in model_params, f"Missing key in 'model_params': '{key}'"

    assert isinstance(model_params.get('feature_dim'), int) and model_params['feature_dim'] > 0, \
        "'model_params.feature_dim' must be a positive integer."
    assert isinstance(model_params.get('embedding_dim'), int) and model_params['embedding_dim'] > 0, \
        "'model_params.embedding_dim' must be a positive integer."
    assert isinstance(model_params.get('output_dim'), int) and model_params['output_dim'] > 0, \
        "'model_params.output_dim' must be a positive integer."
    assert isinstance(model_params.get('n_layers'), int) and model_params['n_layers'] > 0, \
        "'model_params.n_layers' must be a positive integer."
    assert isinstance(model_params.get('n_heads'), int) and model_params['n_heads'] > 0, \
        "'model_params.n_heads' must be a positive integer."
    assert isinstance(model_params.get('dropout'), (float, int)) and 0 <= model_params['dropout'] <= 1, \
        "'model_params.dropout' must be a float or int between 0 and 1."

    # --- Training Parameters ---
    assert 'training_params' in config and isinstance(config['training_params'], dict), "'training_params' section is missing or not a dictionary."
    train_params = config['training_params']
    required_train_keys = ['device', 'batch_size', 'n_epochs', 'patience', 'loss_function', 'learning_rate',
                           'optimizer', 'weight_decay', 'beta1', 'beta2', 'adam_eps']
    for key in required_train_keys:
        assert key in train_params, f"Missing key in 'training_params': '{key}'"

    assert isinstance(train_params.get('device'), str) and (train_params['device'] == 'cpu' or re.match(r'^cuda:\d+$', train_params['device'])), \
        "'training_params.device' must be 'cpu' or 'cuda:X' (e.g., 'cuda:0')."
    assert isinstance(train_params.get('batch_size'), int) and train_params['batch_size'] > 0, \
        "'training_params.batch_size' must be a positive integer."
    assert isinstance(train_params.get('n_epochs'), int) and train_params['n_epochs'] > 0, \
        "'training_params.n_epochs' must be a positive integer."
    assert isinstance(train_params.get('patience'), int) and train_params['patience'] >= -1, \
        "'training_params.patience' must be an integer >= -1."
    assert isinstance(train_params.get('loss_function'), str), "'training_params.loss_function' must be a string."
    assert isinstance(train_params.get('learning_rate'), float) and train_params['learning_rate'] > 0, \
        "'training_params.learning_rate' must be a positive float."
    assert isinstance(train_params.get('optimizer'), str) and train_params['optimizer'] in ['Adam', 'AdamW'], \
        "'training_params.optimizer' must be 'Adam' or 'AdamW'." # Add more if supported
    assert isinstance(train_params.get('weight_decay'), float) and train_params['weight_decay'] >= 0, \
        "'training_params.weight_decay' must be a non-negative float."
    assert isinstance(train_params.get('beta1'), float) and 0 < train_params['beta1'] < 1, \
        "'training_params.beta1' must be a float between 0 and 1."
    assert isinstance(train_params.get('beta2'), float) and 0 < train_params['beta2'] < 1, \
        "'training_params.beta2' must be a float between 0 and 1."
    try:
        adam_eps_float = float(train_params.get('adam_eps'))
        assert adam_eps_float > 0, "'training_params.adam_eps' must be positive after conversion to float."
    except (ValueError, TypeError, AssertionError) as e:
        raise AssertionError(f"'training_params.adam_eps' must be convertible to a positive float. Error: {e}")

    # --- Learning Rate Scheduler (Optional) ---
    if config.get('use_one_cycle_lr', False):
        assert isinstance(config.get('use_one_cycle_lr'), bool), "'use_one_cycle_lr' must be a boolean."
        assert 'one_cycle_lr_params' in config and isinstance(config['one_cycle_lr_params'], dict), \
            "'one_cycle_lr_params' section is missing or not a dictionary when 'use_one_cycle_lr' is true."
        lr_params = config['one_cycle_lr_params']
        required_lr_keys = ['max_lr', 'pct_start', 'anneal_strategy', 'div_factor', 'final_div_factor']
        for key in required_lr_keys:
            assert key in lr_params, f"Missing key in 'one_cycle_lr_params': '{key}'"

        assert isinstance(lr_params.get('max_lr'), float) and lr_params['max_lr'] > 0, \
            "'one_cycle_lr_params.max_lr' must be a positive float."
        assert isinstance(lr_params.get('pct_start'), float) and 0 < lr_params['pct_start'] < 1, \
            "'one_cycle_lr_params.pct_start' must be a float between 0 and 1."
        assert isinstance(lr_params.get('anneal_strategy'), str) and lr_params['anneal_strategy'] in ['cos', 'linear'], \
            "'one_cycle_lr_params.anneal_strategy' must be 'cos' or 'linear'."
        assert isinstance(lr_params.get('div_factor'), float) and lr_params['div_factor'] > 0, \
            "'one_cycle_lr_params.div_factor' must be a positive float."
        assert isinstance(lr_params.get('final_div_factor'), float) and lr_params['final_div_factor'] > 0, \
            "'one_cycle_lr_params.final_div_factor' must be a positive float."

    # --- Logging (Optional) ---
    assert 'log_every_n_steps' in config, "Missing required logging key: 'log_every_n_steps'"
    assert isinstance(config.get('log_every_n_steps'), int) and config['log_every_n_steps'] > 0, \
        "'log_every_n_steps' must be a positive integer."

    if config.get('use_tensorboard', False):
        assert isinstance(config.get('use_tensorboard'), bool), "'use_tensorboard' must be a boolean."
        assert 'tensorboard_params' in config and isinstance(config['tensorboard_params'], dict), \
            "'tensorboard_params' section is missing or not a dictionary when 'use_tensorboard' is true."
        tb_params = config['tensorboard_params']
        assert 'log_dir' in tb_params, "Missing key in 'tensorboard_params': 'log_dir'"
        assert isinstance(tb_params.get('log_dir'), str), "'tensorboard_params.log_dir' must be a string."

    if config.get('use_wandb', False):
        assert isinstance(config.get('use_wandb'), bool), "'use_wandb' must be a boolean."
        assert 'wandb_params' in config and isinstance(config['wandb_params'], dict), \
            "'wandb_params' section is missing or not a dictionary when 'use_wandb' is true."
        wandb_params = config['wandb_params']
        assert 'wandb_project' in wandb_params, "Missing key in 'wandb_params': 'wandb_project'"
        assert isinstance(wandb_params.get('wandb_project'), str), "'wandb_params.wandb_project' must be a string."

    print("Training configuration validation passed.")


def assert_config_inference(config):
    """
    Validates the configuration dictionary for the inference script.
    Focuses on parameters needed to load the model and data for inference.

    Args:
        config (dict): The loaded configuration dictionary.

    Raises:
        AssertionError: If any configuration check fails.
    """
    print("Validating inference configuration...")

    # --- Required Sections ---
    required_sections = ['input_data', 'model_params', 'training_params']
    for section in required_sections:
        assert section in config and isinstance(config[section], dict), \
            f"Required section '{section}' is missing or not a dictionary."
        
    # --- Checkpoint Path ---
    assert 'checkpoint_path' in config, "Missing required key: 'checkpoint_path'"
    assert isinstance(config.get('checkpoint_path'), str), "'checkpoint_path' must be a string."
    assert config['checkpoint_path'] != '', "'checkpoint_path' cannot be an empty string."
    assert config['checkpoint_path'].endswith('.ckpt'), "'checkpoint_path' must end with '.ckpt'."
    assert os.path.isfile(config['checkpoint_path']), f"Checkpoint file '{config['checkpoint_path']}' does not exist."

    # --- Input Data Parameters (subset needed for inference) ---
    input_data = config['input_data']
    required_input_keys = ['root_dir', 'seq_dim', 'num_workers', 'reconstruction_target'] # Essential for loading data/model
    for key in required_input_keys:
        assert key in input_data, f"Missing key in 'input_data': '{key}'"

    assert isinstance(input_data.get('root_dir'), str), "'input_data.root_dir' must be a string."
    assert isinstance(input_data.get('seq_dim'), int) and input_data['seq_dim'] > 0, \
        "'input_data.seq_dim' must be a positive integer."
    assert isinstance(input_data.get('num_workers'), int) and input_data['num_workers'] >= 0, \
        "'input_data.num_workers' must be a non-negative integer."
    assert isinstance(input_data.get('reconstruction_target'), str), "'input_data.reconstruction_target' must be a string."


    # --- Model Parameters (needed to instantiate model architecture) ---
    model_params = config['model_params']
    required_model_keys = ['feature_dim', 'embedding_dim', 'output_dim', 'n_layers', 'n_heads', 'dropout']
    for key in required_model_keys:
        assert key in model_params, f"Missing key in 'model_params': '{key}'"

    assert isinstance(model_params.get('feature_dim'), int) and model_params['feature_dim'] > 0, \
        "'model_params.feature_dim' must be a positive integer."
    assert isinstance(model_params.get('embedding_dim'), int) and model_params['embedding_dim'] > 0, \
        "'model_params.embedding_dim' must be a positive integer."
    assert isinstance(model_params.get('output_dim'), int) and model_params['output_dim'] > 0, \
        "'model_params.output_dim' must be a positive integer."
    assert isinstance(model_params.get('n_layers'), int) and model_params['n_layers'] > 0, \
        "'model_params.n_layers' must be a positive integer."
    assert isinstance(model_params.get('n_heads'), int) and model_params['n_heads'] > 0, \
        "'model_params.n_heads' must be a positive integer."
    assert isinstance(model_params.get('dropout'), (float, int)) and 0 <= model_params['dropout'] <= 1, \
        "'model_params.dropout' must be a float or int between 0 and 1."

    # --- Training Parameters (subset needed for inference, e.g., device, batch_size) ---
    train_params = config['training_params']
    required_train_keys = ['device', 'batch_size'] # Essential for running inference
    for key in required_train_keys:
        assert key in train_params, f"Missing key in 'training_params': '{key}'"

    assert isinstance(train_params.get('device'), str) and (train_params['device'] == 'cpu' or re.match(r'^cuda:\d+$', train_params['device'])), \
        "'training_params.device' must be 'cpu' or 'cuda:X' (e.g., 'cuda:0')."
    assert isinstance(train_params.get('batch_size'), int) and train_params['batch_size'] > 0, \
        "'training_params.batch_size' must be a positive integer (used for inference batching)."

    # --- Inference Parameters ---
    assert 'inference_params' in config and isinstance(config['inference_params'], dict), \
        "'inference_params' section is missing or not a dictionary."
    inference_params = config['inference_params']
    required_inference_keys = ['inference_root_dir', 'inference_dataset_id', 'inference_parts', 'inference_output_path']
    for key in required_inference_keys:
        assert key in inference_params, f"Missing key in 'inference_params': '{key}'"
    assert isinstance(inference_params.get('inference_root_dir'), str), "'inference_params.inference_root_dir' must be a string."
    assert isinstance(inference_params.get('inference_dataset_id'), list), "'inference_params.inference_dataset_id' must be a list."
    assert all(isinstance(x, int) for x in inference_params['inference_dataset_id']), \
        "'inference_params.inference_dataset_id' must contain only integers."
    assert isinstance(inference_params.get('inference_parts'), list) and \
        all(isinstance(x, int) for x in inference_params['inference_parts']), \
        "'inference_params.inference_parts' must be a list of integers."
    assert isinstance(inference_params.get('inference_output_path'), str), \
        "'inference_params.inference_output_path' must be a string."
    assert inference_params['inference_output_path'] != '', \
        "'inference_params.inference_output_path' cannot be an empty string."
    assert os.path.isdir(inference_params['inference_output_path']), \
        f"Inference output path '{inference_params['inference_output_path']}' is not a valid directory."
    assert os.path.exists(inference_params['inference_output_path']), \
        f"Inference output path '{inference_params['inference_output_path']}' does not exist."
    assert os.access(inference_params['inference_output_path'], os.W_OK), \
        f"Inference output path '{inference_params['inference_output_path']}' is not writable."
