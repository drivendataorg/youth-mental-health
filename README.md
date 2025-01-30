[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-prod-public.s3.amazonaws.com/comp_images/top-view-doctor-using-laptop-clipboard.jpg' width='500'>](https://www.drivendata.org/competitions/group/cdc-narratives/)



# Youth Mental Health Narratives

## Goal of the Competition

Suicide is one of the leading causes of death in the United States for 5-24 year-olds. In order to better understand the circumstances around youth suicides and inform potential interventions, researchers and policymakers rely on several datasets. One key dataset is the [National Violent Death Reporting System (NVDRS)](https://wisqars.cdc.gov/about/nvdrs-data/), which has been tracking information about violent deaths since 2003. The NVDRS dataset is based on law enforcement reports, medical examiner and coroner reports, and death certificates. The NVDRS dataset includes both narrative descriptions of each incident, and common factor variables like precipitating events. The process of generating consistent narratives and accurate factor variables is time-consuming and prone to error.

In this challenge, solvers will help CDC extract information from narratives in the NVDRS, improving both the quality and coverage of the NVDRS dataset. **Higher-quality data can enable researchers across the country to better understand and prevent youth suicides on a national scale.**

This challenge had two tracks:

There are two competition tracks, each with its own associated prizes.

1. In the **Automated Abstraction track**, solvers applied machine learning techniques to automate the population of factor variables from NVDRS' narrative text. The algorithms developed will help streamline the process of manual abstraction and data quality control.
2. In the **Novel Variables track**, solvers explored the NVDRS narratives and extract novel variables that could be used to advance youth mental health research.

## What's in this Repository

This repository contains code from winning competitors in the [Youth Mental Health Narratives challenge](https://www.drivendata.org/competitions/group/cdc-narratives/) on DrivenData. Code for all winning solutions are open source under the MIT License.

Solution code for the two tracks can be found in the `automated-abstraction/` and `novel-variables/` subdirectories, respectively. Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Automated Abstraction Track

| Place | Team or User  | Public Score | Private Score | Summary of Model                           |
|-------|---------------|--------------|---------------|--------------------------------------------|
| 1st     | kbrodt | 0.8706       | 0.8660       | Fine-tuned and performed inference with an ensemble of BigBirg and Longformer models using LoRA  |
| 2nd     | D and T | 0.8676       | 0.8650       | Fine-tuned and performed  inference with a weighted ensemble of DeBERTa and Longformer models |
| 3rd     | dylanliu | 0.8690       | 0.8636       | Generated and soft-labeled additional data using Qwen and Mistral, then fine-tuned and performed inference with DeBERTa and Gemma models  |
| 4th     | bbb88 | 0.8655       | 0.8624       | Generated and soft-labeled additional data using Llama, Phi, Mistral, and Yi, then fine-tuned and performed inference with DeBERTa, Gemma, and Llama models  |

### Novel Variables Track

| Place | Team or User  | Summary of Approach  |
|-------|---------------|--------------|
| 1st & midpoint bonus     | verto | Extracted temporal information to determine preceding events and create a time series leading up to a suicide       |
| 2nd & midpoint bonus     | HealthHackers | Extracted temporal information to determine preceding events and create a time series leading up to a suicide       |
| 3rd & midpoint bonus     | UM-ATLAS | Extracted temporal information to determine preceding events and create a time series leading up to a suicide       |
| Midpoint bonus     | jackson5 | Identified panic attacks  |
| Midpoint bonus     | MPWARE | Extracted temporal information to determine preceding events and create a time series leading up to a suicide       |

---

**Winners Blog Post: [https://drivendata.co/blog/youth-mental-health-winners](https://drivendata.co/blog/youth-mental-health-winners)**

**Automated Abstraction Benchmark Blog Post: [https://drivendata.co/blog/automated-abstraction-benchmark](https://drivendata.co/blog/automated-abstraction-benchmark)**