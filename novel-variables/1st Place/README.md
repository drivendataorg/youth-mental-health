# Solution - Youth Mental Health Narratives: Novel Variables

Username: cyong, ichan_verto, hosseinyousefii

## Summary
We extract temporal variables from mental health narratives and use them to create a population health journey. This solution contains 6 parts: Removing notes without temporal aspects, temporal extraction, sentence topic modelling, event log creation and visualization. An ensemble of Flan-T5 models were used at every step with prompt engineering. No training is required for this solution.

# Repo Organization
```
.
├── README.md                <- You are here!
├── src                      <- Folder for your project's source code
├── input                    
    └── features_Z140Hep.csv <- Competition file containing notes and its corresponding features
```

# Setup

1. Install the prerequisities
     - Python version 3.10

2. Install the required python packages

```bash
pip install -r requirements.txt
```

# Hardware

For local development, a Macbook Pro (M1 Pro 32GB RAM) was used. We used `flan-t5-large` for development.

The final solution was run on Azure ML Studio Standard_NC24ads_A100_v4 (80GB vRAM) and with `flan-t5-xl`.

Step 1:
Aggregate and extracted the time-related sentence data, follow with sentence classification for each victim
● Azure ML Studio Standard_NC24ads_A100_v4 (80GB vRAM)
● Inference duration: 2.5 hours

Step 2:
Relative timing word extraction, normalization and classification.
- Azure ML Studio Standard_NC24ads_A100_v4 (80GB vRAM)
-  Inference duration: 1.5 hours

All other steps
- Macbook Pro
- CPU: Apple M1 Pro
- RAM: 32GB
- Inference duration: within 2 hours
- Manually annotating data: 7 hours

# Run inference

Run the following jupyter notebooks in the order below:
```
1. process1.ipynb
2. process2.ipynb
3. process3.ipynb
4. visualize.ipynb
```