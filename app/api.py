from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
from PIL import Image
import io
import os
from dotenv import load_dotenv

from .models.llm_model import LLMModel
from .models.clip_model import CLIPSearchModel
from .models.preference_model import PreferenceModel
from .database.vector_store import VectorStore
from .utils.helpers import (
    save_listings_to_json,
    load_listings_from_json,
    format_listing_for_display,
    ensure_directory_exists,
    validate_listing,
    filter_listings_by_criteria
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="HomeMatch", description="AI-powered real estate matching system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm_model = LLMModel()
clip_model = CLIPSearchModel()
vector_store = VectorStore()
preference_model = PreferenceModel()

@app.post("/generate-listings")
async def generate_listings(num_listings: int = 10):
    """
    Generate sample real estate listings
    """
    listings = llm_model.generate_sample_listings(num_listings)

    # Save listings to JSON file
    ensure_directory_exists("data")
    save_listings_to_json(listings, "data/listings.json")

    # Add listings to vector store
    embeddings = []
    for listing in listings:
        # Create a combined text representation for embedding
        text_for_embedding = f"{listing['description']} {listing['neighborhood_description']}"
        embedding = llm_model.get_embedding(text_for_embedding)
        embeddings.append(embedding)

    vector_store.add_listings(listings, embeddings)

    return {"message": f"Generated {len(listings)} listings", "listings": listings}

@app.post("/search-listings")
async def search_listings(
    preferences: List[str],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    min_bathrooms: Optional[int] = None,
    min_size: Optional[float] = None,
    n_results: int = 5
):
    """
    Search for listings based on buyer preferences
    """
    # Parse buyer preferences using the preference model
    structured_preferences = preference_model.parse_preferences(preferences)

    # Generate search query from structured preferences
    search_query = preference_model.generate_search_query(structured_preferences)

    # Get embedding for the search query
    query_embedding = llm_model.get_embedding(search_query)

    # Search vector store
    matches = vector_store.search_listings(query_embedding, n_results)

    # Filter results based on criteria
    filtered_matches = filter_listings_by_criteria(
        [match["metadata"] for match in matches],
        min_price=min_price,
        max_price=max_price,
        min_bedrooms=min_bedrooms,
        min_bathrooms=min_bathrooms,
        min_size=min_size
    )

    # Generate personalized descriptions
    for listing in filtered_matches:
        personalized_description = llm_model.generate_listing_description(
            listing,
            structured_preferences.__dict__  # Convert dataclass to dict
        )
        listing["personalized_description"] = personalized_description

    return {
        "matches": filtered_matches,
        "structured_preferences": structured_preferences.__dict__,
        "search_query": search_query
    }

@app.post("/image-search")
async def image_search(
    image: UploadFile = File(...),
    n_results: int = 5
):
    """
    Search for similar properties using an image
    """
    # Read and process the uploaded image
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))

    # Get all listings
    all_listings = vector_store.get_all_listings()

    # Add image embeddings to listings
    for listing in all_listings:
        # For now, we'll use the text description for CLIP embedding
        # In a real application, you would use actual property images
        text_for_clip = f"{listing['metadata']['description']} {listing['metadata']['neighborhood_description']}"
        listing["image_embedding"] = clip_model.get_text_embedding(text_for_clip)

    # Search for similar properties
    similar_properties = clip_model.search_similar_properties(
        pil_image,
        all_listings,
        top_k=n_results
    )

    return {"matches": similar_properties}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)