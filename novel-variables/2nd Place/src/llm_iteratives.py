import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch
import re

# Constants and Configuration
HUGGINGFACE_TOKEN = "<Place your token here>"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
INPUT_CSV = r'../data/raw/features_Z140Hep.csv'
OUTPUT_CSV = r'../data/outputs/llm_with_clustering/llm_responses.csv'
EXTRACTED_VARIABLES_FILE = r'../data/outputs/llm_with_clustering/cleaned_variable_list.txt'
QUERY_FOR_LLM = """What are the novel variables can you understand from the given context?"""
MAX_NEW_TOKENS = 512  # Maximum new tokens generated
ROWS_TO_PROCESS = 10 #Number of narrative rows interact with the LLM context.

# Functions

def authenticate_huggingface(token):
    """Authenticate with HuggingFace Hub."""
    login(token=token)

def load_model_and_pipeline(model_id):
    """Load the tokenizer, model, and pipeline."""
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
        max_length=MAX_NEW_TOKENS,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id  # Fix warning by setting pad_token_id
    )
    return tokenizer, pipe

def truncate_prompt_if_needed(tokenizer, context, query, max_length):
    """Truncate context to fit within model input limits."""
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if len(prompt_tokens) > max_length:
        available_length = max_length - len(tokenizer.encode(f"Question:\n{query}", add_special_tokens=False)) - 2
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        truncated_context_tokens = context_tokens[:available_length]
        truncated_context = tokenizer.decode(truncated_context_tokens)
        prompt = f"Context:\n{truncated_context}\n\nQuestion:\n{query}"
    return prompt

def generate_response(pipe, tokenizer, narrative_text, query):
    """Generate a response using the LLM pipeline."""
    prompt = truncate_prompt_if_needed(tokenizer, narrative_text, query, MAX_NEW_TOKENS)
    output = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS
    )
    return output[0]["generated_text"]

def process_dataset(input_csv, tokenizer, pipe, query, rows_to_process=10):
    """Process the dataset and generate LLM responses for narratives."""
    df = pd.read_csv(input_csv)
    results = []
    for idx, row in df.head(rows_to_process).iterrows():
        combined_text = ' '.join([
            str(row.get('NarrativeLE', '')),
            str(row.get('NarrativeCME', ''))
        ]).strip()

        if not combined_text:  # Skip empty narratives
            continue

        response = generate_response(pipe, tokenizer, combined_text, query)
        results.append({
            'Index': idx,
            'Response': response
        })
    return pd.DataFrame(results)

def save_results(df_results, output_csv):
    """Save results to a CSV file."""
    df_results.to_csv(output_csv, index=False)

def extract_variables_from_responses(input_csv, column_name, output_file):
    """Extract bolded variables from responses and save cleaned variables to a file."""
    df = pd.read_csv(input_csv)
    all_variables = []

    for text in df[column_name].dropna():
        # Step 1: Extract bolded variable names
        pattern = r"\*\*(.*?)\*\*"
        bolded_variables = re.findall(pattern, text)

        # Step 2: Convert to snake_case
        def to_snake_case(name):
            return name.lower().replace(" ", "_")

        # Step 3: Apply transformation and extend the list
        all_variables.extend([to_snake_case(var) for var in bolded_variables])

    # Step 4: Remove duplicates and clean
    unique_variables = list(set(all_variables))
    cleaned_variables = [var for var in unique_variables if var and var.strip()]

    # Step 5: Save variables to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in cleaned_variables:
            file.write(f"{item}\n")

    print(f"Extracted {len(cleaned_variables)} unique variables and saved to {output_file}")

def main():
    """Main function to orchestrate the workflow."""
    # Step 1: Authenticate
    authenticate_huggingface(HUGGINGFACE_TOKEN)

    # Step 2: Load Model and Pipeline
    tokenizer, pipe = load_model_and_pipeline(MODEL_ID)

    # Step 3: Process Dataset
    df_results = process_dataset(INPUT_CSV, tokenizer, pipe, QUERY_FOR_LLM,ROWS_TO_PROCESS)

    # Step 4: Save Results
    save_results(df_results, OUTPUT_CSV)
    print(f"LLM responses saved to {OUTPUT_CSV}")

    # Step 5: Extract Variables from LLM Responses
    extract_variables_from_responses(OUTPUT_CSV, 'Response', EXTRACTED_VARIABLES_FILE)

# Run the script
if __name__ == "__main__":
    main()
