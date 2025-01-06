class CFG:
    NUM_WORKERS = 4
    BATCH_SIZE = 1
    MODEL_CONFIGS = [
        {
            'base_model_path': 'assets/gemma2_2b_it', 
            'max_length': 3072,
            'batch_size': 1, 
            'lora_configs': [
                {
                    'path': 'assets/model1_1.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 128, 
                    'alpha': 256,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model1_2.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 128, 
                    'alpha': 256,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model1_3.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 128, 
                    'alpha': 256,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model1_4.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 128, 
                    'alpha': 256,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model1_5.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 128, 
                    'alpha': 256,
                    'weight': 0.29,
                },
            ]
        },
        {
            'base_model_path': 'assets/gemma2_9b_it_8bit', 
            'max_length': 2048,
            'batch_size': 1, 
            'lora_configs': [
                {
                    'path': 'assets/model2_1.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 64, 
                    'alpha': 128,
                    'weight': 1.11,
                },
                {
                    'path': 'assets/model2_2.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 64, 
                    'alpha': 128,
                    'weight': 1.11,
                },
                {
                    'path': 'assets/model2_3.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 64, 
                    'alpha': 128,
                    'weight': 1.11,
                },
                {
                    'path': 'assets/model2_4.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 64, 
                    'alpha': 128,
                    'weight': 1.11,
                },
                {
                    'path': 'assets/model2_5.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 64, 
                    'alpha': 128,
                    'weight': 1.11,
                },

                {
                    'path': 'assets/model3_1.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 32, 
                    'alpha': 64,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model3_2.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 32, 
                    'alpha': 64,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model3_3.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 32, 
                    'alpha': 64,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model3_4.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 32, 
                    'alpha': 64,
                    'weight': 0.29,
                },
                {
                    'path': 'assets/model3_5.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 32, 
                    'alpha': 64,
                    'weight': 0.29,
                },

                {
                    'path': 'assets/model4_1.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 48, 
                    'alpha': 128,
                    'weight': 1.13,
                },
                {
                    'path': 'assets/model4_2.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 48, 
                    'alpha': 128,
                    'weight': 1.13,
                },
                {
                    'path': 'assets/model4_3.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 48, 
                    'alpha': 128,
                    'weight': 1.13,
                },
                {
                    'path': 'assets/model4_4.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 48, 
                    'alpha': 128,
                    'weight': 1.13,
                },
                {
                    'path': 'assets/model4_5.pth',
                    'model_type': 1,
                    'pad_left': False,
                    'rank': 48, 
                    'alpha': 128,
                    'weight': 1.13,
                },
            ]
        },      
    ]
    DATA_PATH = 'data/test_features.csv'
    SUBMISSION_PATH = 'submission.csv'
    SEED = 42 
    #LORA_MODULES = ['o_proj', 'v_proj']
