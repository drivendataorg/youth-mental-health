MODEL_CFGS = [
    {
        'base_model_path': 'assets/gemma2_2b_it', 
        'max_length': 2560,
        'batch_size': 1, # formerly 8
        'seed': 58,
        'cv_seed': 46,
        'rank': 128, 
        'alpha': 256,
        'save_name': 'model1'
    },
    {
        'base_model_path': 'assets/gemma2_9b_it', 
        'max_length': 1280,
        'batch_size': 1, # formerly 8
        'seed': 2048,
        'cv_seed': 42,
        'rank': 64, 
        'alpha': 128,
        'save_name': 'model2'
    },
    {
        'base_model_path': 'assets/gemma2_9b_it', 
        'max_length': 1280,
        'batch_size': 1, # formerly 8
        'seed': 43,
        'cv_seed': 42,
        'rank': 32, 
        'alpha': 64,
        'save_name': 'model3'
    },
    {
        'base_model_path': 'assets/gemma2_9b_it', 
        'max_length': 2048,
        'batch_size': 1, # formerly 4 
        'seed': 214,
        'cv_seed': 554,
        'rank': 48, 
        'alpha': 128,
        'save_name': 'model4'
    },
]