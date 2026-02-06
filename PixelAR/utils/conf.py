def setup_default_config_values(config):
    # check if an argument is absent, if so add default
    if 'predict_eoi_token' not in config['model']:
        print("setting model.predict_eoi_token to False")
        config['model']['predict_eoi_token'] = False

    if 'encoder_block_causal' not in config['model']:
        print("setting model.encoder_block_causal to False")
        config['model']['encoder_block_causal'] = False

    if 'embedding_type' not in config['model']:
        print("setting model.embedding_type to None")
        config['model']['embedding_type'] = None

    if 'uuid' not in config:
        print("setting uuid to None")
        config['uuid'] = None
        
    if 'use_ca_rope' not in config['training']:
        print("setting training.use_ca_rope to False")
        config['training']['use_ca_rope'] = False
        
    return config
