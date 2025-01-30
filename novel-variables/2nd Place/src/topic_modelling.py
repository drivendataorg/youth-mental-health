import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.cli import download
import string, re
from sentence_transformers import SentenceTransformer
import numpy as np
from bertopic import BERTopic

# Load Sentence Transformer model
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# File paths
INPUT_CSV = r'../data/raw/features_Z140Hep.csv'
OUTPUT_CSV_LE = r'../data/outputs/topic_modelling/output_with_topics_LE.csv'
OUTPUT_CSV_CME = r'../data/outputs/topic_modelling/output_with_topics_CME.csv'

# Pre-defined replacement dictionary
REPLACEMENT_DICT = {
    'v': 'victim',
    'xxxx': '',
    'vs': "victim's",
    'xx17': '',
    'v1': 'victim one',
    'v2': 'victim two',
    's': '',
}

# Regex pattern for punctuation removal
PUNCTUATION_PATTERN = re.compile(r'[\.,!?;:\-\']')

### Functions

def load_data(input_csv):
    """ Load CSV data. """
    return pd.read_csv(input_csv)

def normalize_text(text):
    """ Normalize text: lowercasing, removing punctuation, stopwords, and lemmatization. """
    text = text.lower()
    text = PUNCTUATION_PATTERN.sub(' ', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not (token.is_punct or token.is_space or token.is_stop)]
    return ' '.join(tokens)

def replace_tokens(text, replacement_dict):
    """ Replace specific tokens based on a replacement dictionary. """
    doc = nlp(text)
    modified_tokens = [replacement_dict.get(token.lemma_.lower(), token.text) for token in doc]
    return ' '.join(modified_tokens)

def tokenize_text(text):
    """ Tokenize text into individual words. """
    doc = nlp(text)
    return [token.text for token in doc]

def get_embedding(text, model):
    """ Generate embeddings for a given text using a Sentence Transformer model. """
    return model.encode(text)

def process_column(df, column_name, replacement_dict, model):
    """ Process a column by normalizing, replacing tokens, and creating embeddings. """
    df[f'normalized_{column_name}'] = df[column_name].apply(normalize_text)
    df[f'modified_{column_name}'] = df[f'normalized_{column_name}'].apply(lambda x: replace_tokens(x, replacement_dict))
    df[f'tokens_{column_name}'] = df[f'modified_{column_name}'].apply(tokenize_text)
    df[f'embedding_{column_name}'] = df[f'modified_{column_name}'].apply(lambda x: get_embedding(x, model))
    return df

def apply_bertopic(df, column_name, output_csv):
    """ Apply BERTopic on a column and save topic results. """
    embeddings = np.array(df[f'embedding_{column_name}'].tolist())
    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(df[f'modified_{column_name}'], embeddings)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_csv, index=False)
    print(f"Topics saved to {output_csv}")

### Main Execution
if __name__ == "__main__":
    
    # Load Spacy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading 'en_core_web_sm' model...")
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Load data
    df = load_data(INPUT_CSV)

    # Process NarrativeLE column
    df = process_column(df, 'NarrativeLE', REPLACEMENT_DICT, model)
    apply_bertopic(df, 'NarrativeLE', OUTPUT_CSV_LE)

    # Process NarrativeCME column
    df = process_column(df, 'NarrativeCME', REPLACEMENT_DICT, model)
    apply_bertopic(df, 'NarrativeCME', OUTPUT_CSV_CME)

    print("Processing completed!")
