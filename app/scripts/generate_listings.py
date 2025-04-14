import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.llm_model import LLMModel
from database.vector_store import VectorStore
from utils.helpers import save_listings_to_json, ensure_directory_exists

def generate_and_store_listings(num_listings: int = 10):
    """
    Generate sample listings and store them in both JSON and vector database
    """
    # Initialize models
    llm_model = LLMModel()
    vector_store = VectorStore()

    print(f"Generating {num_listings} sample listings...")
    listings = llm_model.generate_sample_listings(num_listings)

    # Save to JSON
    ensure_directory_exists("data")
    save_listings_to_json(listings, "data/listings.json")
    print("Saved listings to JSON file")

    # Generate embeddings and store in vector database
    print("Generating embeddings and storing in vector database...")
    embeddings = []
    for listing in listings:
        text_for_embedding = f"{listing['description']} {listing['neighborhood_description']}"
        embedding = llm_model.get_embedding(text_for_embedding)
        embeddings.append(embedding)

    vector_store.add_listings(listings, embeddings)
    print("Successfully stored listings in vector database")

    return listings

if __name__ == "__main__":
    num_listings = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    generate_and_store_listings(num_listings)