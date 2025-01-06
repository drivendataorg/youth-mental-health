### Training
1. Install all the required packages: 
`pip install -r requirements.txt`

2. Create 5 folds csv file: 
`python create_folds.py`

3. Configure `cfg.py` file for training parameters
I used 'column_dict' and 'column_manual_dict' in cfg.DEFINE to create 2 different prompt templates.
To train Longformer model, remember to use correct 'collate' function in dataset.py (line 97).

4. Train: 
`python train.py`

5. After we have 5 folds trained models, using them to create out-of-fold predictions and calculate local CV: 
`python predict_5folds.py`

6. Optimize threshold for each binary variable and weights of ensemble models: 
`python optimize_thresh.py`

### Data structure

```
data
├── train_features.csv
├── train_labels.csv
├── smoke_test_features.csv
├── smoke_test_labels.csv
├── submission_format.csv
proc_data
├── train_5folds.csv
src
├── saved_models/
├── logs/
├── predictions/
├── cfg.py
├── create_folds.py
├── dataset.py
├── helpers.py
├── model.py
├── ...
```