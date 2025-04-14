import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import uuid

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
        self.collection = self.client.get_or_create_collection(
            name="real_estate_listings",
            metadata={"hnsw:space": "cosine"}
        )

    def add_listings(self, listings: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Add listings to the vector store with their embeddings
        """
        if not listings or not embeddings:
            print("Warning: No listings or embeddings to add")
            return

        if len(listings) != len(embeddings):
            raise ValueError(f"Number of listings ({len(listings)}) does not match number of embeddings ({len(embeddings)})")

        # Generate unique IDs for each listing
        ids = [str(uuid.uuid4()) for _ in range(len(listings))]
        documents = [str(listing) for listing in listings]
        metadatas = listings

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search_listings(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar listings using a query embedding
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        return [
            {
                "id": id,
                "metadata": metadata,
                "distance": distance
            }
            for id, metadata, distance in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def update_listing(self, listing_id: str, new_metadata: Dict[str, Any], new_embedding: List[float]):
        """
        Update an existing listing
        """
        self.collection.update(
            ids=[listing_id],
            embeddings=[new_embedding],
            metadatas=[new_metadata],
            documents=[str(new_metadata)]
        )

    def delete_listing(self, listing_id: str):
        """
        Delete a listing from the vector store
        """
        self.collection.delete(ids=[listing_id])

    def get_all_listings(self) -> List[Dict[str, Any]]:
        """
        Retrieve all listings from the vector store
        """
        results = self.collection.get()
        return [
            {
                "id": id,
                "metadata": metadata,
                "document": document
            }
            for id, metadata, document in zip(
                results["ids"],
                results["metadatas"],
                results["documents"]
            )
        ]