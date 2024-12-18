import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import kagglehub


# Load Dataset
def load_dataset(file_path):
    """
    Load the news category dataset.
    Args:
        file_path (str): Path to the dataset CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_json(file_path, lines=True)
    data['text'] = data['headline'] + " " + data['short_description']
    return data

# Text Preprocessing
def preprocess_text(text):
    """
    Preprocess text: tokenize, remove stop words, and lemmatize.
    Args:
        text (str): Raw text.
    Returns:
        str: Preprocessed text.
    """

    if pd.isna(text):  # Check for NaN values
        return ""  # Replace NaN with an empty string
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Split Dataset into Subsets
def split_dataset(data, num_clients=5):
    """
    Split the dataset into subsets for synthetic clients.
    Args:
        data (pd.DataFrame): Dataset with categories.
        num_clients (int): Number of subsets (clients).
    Returns:
        dict: Subsets of data mapped to client IDs.
    """
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    n_portions = num_clients
    portion_data = []
    for i in range(n_portions):
        # Stratified split to ensure category distribution in each portion
        train, _ = train_test_split(df, test_size=0.9, stratify=df['category'])
        portion_data.append(train)

    company_data = {f"Company {i+1}": portion for i, portion in enumerate(portion_data)}
    return company_data

# Assign to Hypothetical Companies
def save_subsets_to_files(subsets, output_dir='synthetic_clients'):
    """
    Save each subset as a separate CSV file for simulation.
    Args:
        subsets (dict): Subsets of data for each client.
        output_dir (str): Directory to save files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for client, data in subsets.items():
        file_path = os.path.join(output_dir, f"{client}.csv")
        data.to_csv(file_path, index=False)
        print(f"Saved {client} data to {file_path}")


#Check if data file exists
#Download data file
data = load_dataset("/Users/bacemtayeb/Downloads/project/data/News_Category_Dataset_v3.json")
data['text'] = data['headline'] + " " + data['short_description']
data["preprocessed_text"] = data["text"].apply(preprocess_text)

subsets = split_dataset(data[["processed_text","category"]], num_clients=5)
save_subsets_to_files(subsets)