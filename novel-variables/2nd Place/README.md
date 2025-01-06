


# Solution - Youth Mental Health Narratives: Novel Variables (https://www.drivendata.org/competitions/296/cdc-novel-variables/)


Author: Dinuja Willigoda Liyanage, Lasantha Ranwala, Benjamin Ung, Poorna Fernando 
Username: dinuja, lasantha13, ben.ung, Poorna_F
License: MIT

# Summary
**Data Preprocessing:**
The narrative text data from the NVDRS was preprocessed using spaCy, which included:
 - Tokenization. 
 - Removal of stop words. 
 - Lemmatization. 
 - Handling linguistic inconsistencies for text standardization.
Preprocessed text was normalized into a string format for embedding and subsequent analysis.

**Proposed Architecture:**
Three main approaches were used for extracting variables:
1.	*Topic Modeling Pathway:*
Employed RoBERTa and all-MiniLM-L6-v2 for text embeddings. These models were selected over domain-specific models (e.g., ClinicalBERT, BioBERT) due to the general-language nature of the suicide narratives.
BERTopic was used to generate 52 distinct topics, which were validated and refined by medical experts into clinically relevant variables.

2.	*LLM Response Clustering Pathway:*
Utilized LLAMA-3.2-3B-Instruct for extracting variables through carefully designed prompts.
Processed text embeddings using SBERT (all-MiniLM-L6-v2) and clustered them with DBSCAN, which effectively identified 102 clusters of variable-related keywords.


3.	*Combined Pathway and Ensemble Approach:*
A unified approach integrated Topic Modeling and LLM Response outputs: Word clusters and variables from the Topic Modeling pathway informed chunk creation for similarity search. FAISS was employed to efficiently retrieve semantically relevant text chunks.
The combined pathway leveraged the strengths of both approaches, enhancing accuracy and robustness.

**Postprocessing and Validation:**
Outputs from all pathways were rigorously validated by two medical doctors to ensure clinical relevance and minimize bias:
Dual-review and consensus determination were applied for finalizing novel variables.
Noise points and outliers from clustering were analyzed separately, ensuring no significant information was overlooked.

# Repo Organisation
```
.
├── README.md                  	<- You are here!
├── User-Guide-v1.pdf    		<- Reference for the solution documentation	(PDF Document)
├── User-Guide-v1.docx 	       	<- Reference for the solution documentation (WordDocument)
├── data						<- Solution's data
    ├── outputs     			<- Outputs from approaches
    └── raw      				<- The original, immutable data dump.
├── src                        	<- Solution's source codes
    ├── topic_modelling.py      <- Codebase for topic modeling approach
    └── llm_iteratives.py      	<- Codebase for llm iterative approach
    └── llm_faiss.py           	<- Codebase for llm and topic modeling approach with faiss
├── requirements.txt			<- List of python packages that is used
├── Makefile					<- Makefile with commands like `make requirements`
```
# Environment Setup
	1.Install the prerequisites

| Name  | Version |
|--|--|
|Python  | 3.12.5 |
|CUDA  |12.4  |
	2.Create a virtual environment.
 
		On Windows:
			Step 01.Open Command Prompt: Press `Win + R`, type `cmd`, and hit Enter.
			
			Step 02.Navigate to your project folder: cd path\to\your\project
			
			Step 03.Create the virtual environment: python -m venv venv_name
			Replace `venv_name` with your desired environment name.
			
			Step 04.Activate the virtual environment: venv_name\Scripts\activate
			You should see `(venv_name)` at the start of your command prompt, indicating the virtual environment is active.
			
			Step 05.Deactivate the virtual environment: deactivate
		
		On Linux:
			Step 01.Open a terminal
   			
      		Step 02.Install python3-venv (if not already installed): sudo apt install python3-venv
			
			Step 03.Navigate to your project folder: cd path\to\your\project

			Step 04.Create the virtual environment: python3 -m venv venv_name
			Replace `venv_name` with your desired environment name.
			
			Step 05.Activate the virtual environment: source venv_name/bin/activate
			You should see `(venv_name)` at the start of your terminal prompt, indicating the virtual environment is active.
			
   			Step 06.Deactivate the virtual environment: deactivate
			
			
	3.Install the required python packages using pip install -r requirements.txt






# Hardware and Time

 - CPU: Intel Core i5-14400F  
 - RAM: 32 GB RAM 3600Mhz DDR4 
 - GPU: RTX 4070 GPU (12 GB VRAM) 
 - OS:  Windows 11

| **Pathway**                   | **Task**                            | **Time Taken**       | **Per Record Time** | **Results**                  |
|-------------------------------|-------------------------------------|----------------------|---------------------|------------------------------|
| **Topic Modelling Pathway**   | Preprocessing                       | 5 minutes            | 0.075 seconds       | 52 topics generated          |
|                               | Embedding Generation                | 12 minutes           | 0.18 seconds        |                              |
|                               | Topic Modelling                     | 6 minutes            | 0.09 seconds        |                              |
|                               | **Total**                           | **23 minutes**       |                     |                              |
| **LLM Clustering Pathway**    | LLM Inference                       | 12 hours             | 10.8 seconds        | 102 clusters identified      |
|                               | Clustering (DBSCAN)                 | 10 minutes           | 0.15 seconds        |                              |
|                               | **Total**                           | **12 hours 10 mins** |                     |                              |
| **Combined Pathway**          | LLM Inference (Optimized)           | 15-30 minutes        | -                   | 16 LLM responses identified  |
|                               | Semantic Search/Similarity (FAISS)  | 15 minutes           | -                   | 45 novel variables identified|
|                               | **Total**                           | **1 hour**           |                     |                              |



# Acknowledgement
This project makes use of the following tools and models:

- **LLAMA Models by Meta**  
  LLAMA models are developed by Meta and hosted on Hugging Face.  
  Link: [https://ai.meta.com/llama/](https://ai.meta.com/llama/)

- **Hugging Face Transformers Library**  
  Hugging Face provides tools to work with state-of-the-art language models.  
  Link: [https://huggingface.co](https://huggingface.co)

- **Sentence Transformers - all-MiniLM-L6-v2**  
  Developed by UKPLab and hosted on Hugging Face.  
  Link: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

