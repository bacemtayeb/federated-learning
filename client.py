import flwr as fl
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Process
import time

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

# Client class for federated learning
class NewsClient(fl.client.NumPyClient):

    def __init__(self, local_data):
        """
        Initialize the client with local data, vectorizer, and topic model.
        """
        local_data['text'] = local_data['text'].fillna("").astype(str).str.strip()
        local_data = local_data[local_data['text'] != ""]

        all_texts_processed = [preprocess_text(text) for text in local_data['text']]
        dictionary = Dictionary(all_texts_processed)
        corpus = [dictionary.doc2bow(text) for text in all_texts_processed]
        config = {
            'corpus': corpus,   # Corpus in BoW format
            'dictionary': dictionary  # Gensim dictionary mapping word IDs to words
            }
      
        self.all_texts_processed = all_texts_processed
        self.dictionary = dictionary
        self.corpus = corpus
        # Create the config dictionary

        self.config = config

        vectorizer = TfidfVectorizer(max_features=1000)
        self.local_data = vectorizer.fit_transform(local_data['text'])
        self.vectorizer = vectorizer
        self.model = LatentDirichletAllocation(n_components=5, random_state=42)

        if self.local_data.shape[0] == 0:
            raise ValueError("No valid data for training. The dataset is empty after preprocessing.")
        self.model.fit(self.local_data)

    def get_parameters(self):
        """
        Return the topic-word distribution parameters (model components).
        """
        if hasattr(self.model, 'components_'):
            return [param.flatten() for param in self.model.components_]
        else:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

    def set_parameters(self, parameters):
        """
        Set the parameters (topic-word distributions) of the LDA model.
        """
        for param, new_param in zip(self.model.components_, parameters):
            param[:] = new_param.reshape(param.shape)

    def fit(self, parameters, config):
        """
        Fit the local data using the topic model (LDA).
        """
        self.set_parameters(parameters)
        self.model.fit(self.local_data)
        return self.get_parameters(), self.local_data.shape[0], {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model using custom coherence score.
        """
        # Set the model parameters
        self.set_parameters(parameters)
        
        # Get the topic-word distribution matrix (components_ for LDA)
        topics = self.model.components_  # Shape: (n_topics, n_words)
        
        # Compute coherence score using custom calculation
        coherence_score = self.custom_coherence_calculation(topics)
        
        # Return coherence score only, and include number of parameters for federated learning
        # `accuracy` is not needed unless you're doing classification-style evaluation.
        return coherence_score, len(parameters), {}

    def custom_coherence_calculation(self, topics):
        """
        A custom coherence calculation method, e.g., cosine similarity between topic-word distributions.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute cosine similarity between each pair of topics
        similarity_matrix = cosine_similarity(topics)
        
        # Exclude the diagonal (self-similarity) and compute the average similarity
        num_topics = similarity_matrix.shape[0]
        coherence_score = (similarity_matrix.sum() - num_topics) / (num_topics * (num_topics - 1))
        
        return coherence_score

def load_data_from_csv(directory):
    """
    Load the data for each client from CSV files in the directory.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    company_data = {f"{os.path.splitext(f)[0]}": pd.read_csv(os.path.join(directory, f)) for f in csv_files}
    return company_data

# Function to run each client process
def run_client(company_name, company_data):
    print(f"Starting local training for: {company_name}")
    company_data = company_data.sample(frac=0.25, random_state=42)
    client = NewsClient(local_data=company_data)
    fl.client.start_numpy_client("localhost:8080", client=client)
    print(f"Local training for {company_name} completed.")

# This will run the simulation
def run_simulation(clients_data):
    processes = []
    for company_name, company_data in clients_data.items():
        p = Process(target=run_client, args=(company_name, company_data))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    company_data = load_data_from_csv("synthetic_clients")
    run_simulation(company_data)
