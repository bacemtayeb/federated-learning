import flwr as fl
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import Process

# Custom aggregation logic in fit
def fit(parameters, config):
    print(f"Starting fit round...")
    averaged_parameters = np.mean(parameters, axis=0)
    return averaged_parameters, len(parameters), {}

def evaluate_model_on_server(parameters, config):
    """
    Evaluate the model using custom coherence score on the server.
    """
    print("Evaluating model using custom coherence...")
    
    # Here, parameters are the topic-word distributions from the clients
    topics = parameters  # Parameters are topic-word distributions
    
    # Compute coherence score using the same method as on the client
    coherence_score = custom_coherence_calculation(topics)
    
    print(f"Custom coherence score for current round: {coherence_score}")
    
    # Return the coherence score and the number of parameters (if needed)
    # Note: We return an empty dictionary for metrics unless other metrics are used.
    return coherence_score, len(parameters), {}

def custom_coherence_calculation(topics):
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


# Start the Flower server without on_fit_round_start
def start_server():
    print("Starting Flower server on localhost:8080...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.001,
        fraction_eval=0.001,
        min_fit_clients=1,
        min_available_clients=1,
    )
    fl.server.start_server(server_address="localhost:8080", strategy=strategy)

# Function to run the server process
def run_server():
    p = Process(target=start_server)
    p.start()
    p.join()

if __name__ == "__main__":
    run_server()
