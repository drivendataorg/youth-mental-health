import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
from huggingface_hub import login
import os

# 1. Authentication
def authenticate_huggingface(token):
    """Authenticate Hugging Face Hub."""
    login(token=token)

# 2. Initialize LLM Model and Pipeline
def initialize_llm(model_id):
    """Initialize tokenizer and LLM model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        device_map="auto",
    )
    return tokenizer, pipe

# 3. Load Dataset
def load_dataset(file_path):
    """Load CSV dataset into a DataFrame."""
    return pd.read_csv(file_path)

# 4. Initialize Embedding Model
def initialize_embedding_model():
    """Load SentenceTransformer model."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 5. Text Chunking Function
def chunk_text(text, tokenizer, max_tokens):
    """Split text into smaller chunks respecting max token limits."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
    return chunks

# 6. Build FAISS Index
def build_faiss_index(embedding_model, df_chunks):
    """Compute embeddings and build FAISS index."""
    embeddings = embedding_model.encode(df_chunks['ChunkText'].tolist(), convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# 7. Retrieve Relevant Text
def retrieve_relevant_texts(query, embedding_model, index, df_chunks, df, top_k=5):
    """Retrieve relevant text chunks and their source narratives."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    
    relevant_chunks = df_chunks.iloc[indices[0]]
    original_indices = relevant_chunks['OriginalIndex'].unique()
    context_list = []
    
    for _, row in df.loc[original_indices].iterrows():
        combined_text = ' '.join([
            str(row.get('NarrativeLE', '')),
            str(row.get('NarrativeCME', ''))
        ])
        context_list.append(combined_text)
    
    return context_list

# 8. Generate Response
def generate_response(queryForLLM, queryForSS, tokenizer, pipe, embedding_model, index, df_chunks, df):
    """Generate a response using retrieved context."""
    relevant_texts = retrieve_relevant_texts(queryForSS, embedding_model, index, df_chunks, df)
    context = "\n".join(relevant_texts)
    prompt = f"Context:\n{context}\n\nQuestion:\n{queryForLLM}"
    
    # Adjust token limits if needed
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    max_length = 1024
    
    if len(prompt_tokens) > max_length:
        available_length = max_length - len(tokenizer.encode(f"Question:\n{queryForLLM}", add_special_tokens=False)) - 2
        truncated_context_tokens = tokenizer.encode(context, add_special_tokens=False)[:available_length]
        truncated_context = tokenizer.decode(truncated_context_tokens)
        prompt = f"Context:\n{truncated_context}\n\nQuestion:\n{queryForLLM}"
    
    output = pipe(prompt, max_new_tokens=1024)
    return output[0]["generated_text"]

# 9. Save Generated Response
def save_response_to_file(queryForLLM, queryForSS, filename, tokenizer, pipe, embedding_model, index, df_chunks, df):
    """Generate a response and save it to a file."""
    response = generate_response(queryForLLM, queryForSS, tokenizer, pipe, embedding_model, index, df_chunks, df)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response)
    print(f"Response saved to {filename}")


# Main Execution
if __name__ == "__main__":
    # Configuration
    HF_TOKEN = "<Place your token here>"
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    INPUT_CSV = r'../data/raw/features_Z140Hep.csv'
    OUTPUT_DIR = r'../data/outputs/llm_with_faiss'  # Directory to save response files
    queryForLLM = "What are the novel variables you can understand from the given context?"  # Static queryForLLM
    
    # List of `queryForSS` Search Queries that generated by topic modeling and manual verification
    queryForSS_list = [
        "cell jail inmate prison",
        "detail rifle mention home",
        "disorder hang victim depression ideation",
        "intoxication diphenhydramine nitrite",
        "jump bridge blunt vehicle",
        "medication dad prior diagnose anxiety",
        "overdose amitriptyline",
        "pill overdose medication",
        "previous belt mod",
        "school migraine",
        "send father victim text phone",
        "treatment depression",
        "tree hang rope find",
        "vehicle burn carbon monoxide charcoal",
        "wife victim gun argument",
        "wound head gunshot vehicle",
    ]    

    # Authenticate and Initialize
    authenticate_huggingface(HF_TOKEN)
    tokenizer, pipe = initialize_llm(MODEL_ID)
    embedding_model = initialize_embedding_model()
    embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Load Data and Process
    df = load_dataset(INPUT_CSV)
    max_tokens = embedding_model.get_max_seq_length() - 2
    
    # Create Chunks
    chunks_data = []
    for idx, row in df.iterrows():
        combined_text = ' '.join([
            str(row.get('NarrativeLE', '')),
            str(row.get('NarrativeCME', ''))
        ])
        if combined_text.strip():
            chunks = chunk_text(combined_text, embedding_tokenizer, max_tokens)
            for chunk in chunks:
                chunks_data.append({'OriginalIndex': idx, 'ChunkText': chunk})
    
    df_chunks = pd.DataFrame(chunks_data)
    
    # Build FAISS Index
    index, _ = build_faiss_index(embedding_model, df_chunks)

    

    # Create Output Directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop through `queryForSS` and save responses
    for i, queryForSS in enumerate(queryForSS_list):
        filename = os.path.join(OUTPUT_DIR, f"response_{i+1}.txt")
        save_response_to_file(queryForLLM, queryForSS, filename, tokenizer, pipe, embedding_model, index, df_chunks, df)
