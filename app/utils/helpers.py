from typing import List, Dict, Any
import json
import os
from pathlib import Path

def save_listings_to_json(listings: List[Dict[str, Any]], output_file: str):
    """
    Save listings to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(listings, f, indent=2)

def load_listings_from_json(input_file: str) -> List[Dict[str, Any]]:
    """
    Load listings from a JSON file
    """
    with open(input_file, 'r') as f:
        return json.load(f)

def format_listing_for_display(listing: Dict[str, Any], include_similarity: bool = False) -> str:
    """
    Format a listing for display
    """
    output = []
    output.append(f"🏠 {listing.get('neighborhood', 'N/A')}")
    output.append(f"💰 Price: ${listing.get('price', 0):,}")
    output.append(f"🛏️ Bedrooms: {listing.get('bedrooms', 'N/A')}")
    output.append(f"🚿 Bathrooms: {listing.get('bathrooms', 'N/A')}")
    output.append(f"📐 Size: {listing.get('house_size', 0):,} sqft")

    if 'description' in listing:
        output.append("\n📝 Description:")
        output.append(listing['description'])

    if 'neighborhood_description' in listing:
        output.append("\n🏘️ Neighborhood:")
        output.append(listing['neighborhood_description'])

    if include_similarity and 'similarity_score' in listing:
        output.append(f"\n📊 Match Score: {listing['similarity_score']:.2%}")

    return "\n".join(output)

def ensure_directory_exists(directory: str):
    """
    Create directory if it doesn't exist
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def validate_listing(listing: Dict[str, Any]) -> bool:
    """
    Validate that a listing contains all required fields
    """
    required_fields = [
        'neighborhood',
        'price',
        'bedrooms',
        'bathrooms',
        'house_size',
        'description',
        'neighborhood_description'
    ]

    return all(field in listing for field in required_fields)

def filter_listings_by_criteria(
    listings: List[Dict[str, Any]],
    min_price: float = None,
    max_price: float = None,
    min_bedrooms: int = None,
    min_bathrooms: int = None,
    min_size: float = None
) -> List[Dict[str, Any]]:
    """
    Filter listings based on basic criteria
    """
    filtered_listings = listings.copy()

    if min_price is not None:
        filtered_listings = [l for l in filtered_listings if l['price'] >= min_price]
    if max_price is not None:
        filtered_listings = [l for l in filtered_listings if l['price'] <= max_price]
    if min_bedrooms is not None:
        filtered_listings = [l for l in filtered_listings if l['bedrooms'] >= min_bedrooms]
    if min_bathrooms is not None:
        filtered_listings = [l for l in filtered_listings if l['bathrooms'] >= min_bathrooms]
    if min_size is not None:
        filtered_listings = [l for l in filtered_listings if l['house_size'] >= min_size]

    return filtered_listings