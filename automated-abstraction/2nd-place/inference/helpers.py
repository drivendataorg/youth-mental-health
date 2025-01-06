import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score


def inference_fn(test_loader, model, config, batch_size, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        num_sample = len(inputs["input_ids"])
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            y_preds = y_preds[:, config.yes_id] - y_preds[:, config.no_id]
            # print(y_preds.shape)
            y_preds = y_preds.view(num_sample, -1)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


def inference_flan_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        with torch.no_grad():
            y_preds = model.generate(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device),
                                    return_dict_in_generate=True, 
                                    output_scores=True)
            y_preds = y_preds.scores[1:78:2]

            # Compute differences and transform shape in one step
            differences = torch.stack([(scores[:, 4273] - scores[:, 150]).unsqueeze(-1) for scores in y_preds])

            # Combine squeezing and permuting
            y_preds = differences.squeeze(-1).permute(1, 0)  # Shape becomes (batch_size, 39)
        preds.append(y_preds.to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions



def convert_predictions(preds, threshold, column_dict):
    binary_preds = preds[:, :21]
    binary_preds = (binary_preds >= threshold).astype(int)
    # binary_preds[binary_preds >= threshold] = 1
    # binary_preds[binary_preds < threshold] = 0
    binary_preds = binary_preds.astype(int)
    
    il_preds = preds[:, 21:27]
    il_preds = np.argmax(il_preds, axis=1) + 1
    wt_preds = preds[:, 27:]
    wt_preds = np.argmax(wt_preds, axis=1) + 1

    il_preds = np.expand_dims(il_preds, axis=1).astype(int)
    wt_preds = np.expand_dims(wt_preds, axis=1).astype(int)
    final_preds = np.concatenate((np.concatenate((binary_preds, il_preds), axis=1), wt_preds), axis=1)

    return final_preds

def average_f1(predictions, labels):
    """Score a set of predictions using the competition metric. F1 is averaged
    across all target variables. For categorical variables, micro-averaged
    F1 score is used.

    Args:
        predictions (pd.DataFrame): Dataframe of predictions, with one column
            for each target variable. The index should be the uid.
        labels (pd.DataFrame): Dataframe of ground truth values, with one column
            for each target variable. The index should be the uid.
    """
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    # Check that column order and row order are the same
    assert (predictions.columns == labels.columns).all()
    assert (predictions.index == labels.index).all()

    # All values should be integers
    assert (predictions.dtypes == int).all()

    CATEGORICAL_VARS = ["InjuryLocationType", "WeaponType1"]
    BINARY_VARS = np.setdiff1d(labels.columns, CATEGORICAL_VARS)

    # Calculate F1 score averaged across binary variables
    binary_f1 = f1_score(
        labels[BINARY_VARS],
        predictions[BINARY_VARS],
        average="macro",
    )
    f1s = [binary_f1]
    # Calculate F1 score for each categorical variable
    for cat_col in CATEGORICAL_VARS:
        f1s.append(f1_score(labels[cat_col], predictions[cat_col], average="micro"))

    return np.average(f1s, weights=[len(BINARY_VARS), 1, 1])