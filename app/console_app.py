from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from models.preference_model import PreferenceModel
from models.llm_model import LLMModel
from utils.helpers import format_listing_for_display, filter_listings_by_criteria
from database.vector_store import VectorStore

def print_welcome():
    """Print welcome message and instructions"""
    print("\n Welcome to HomeMatch Console!")
    print("We'll help you find your perfect home by understanding your preferences.")
    print("\nPlease answer the following questions to help us understand what you're looking for.")
    print("Feel free to be as detailed as you'd like in your answers.\n")

def collect_additional_filters() -> Dict[str, Any]:
    """Collect additional numeric filters from the user"""
    filters = {}

    try:
        print("\nLet's set some specific criteria (press Enter to skip):")

        min_price = input("Minimum price ($): ")
        if min_price.strip():
            filters['min_price'] = float(min_price)

        max_price = input("Maximum price ($): ")
        if max_price.strip():
            filters['max_price'] = float(max_price)

        min_bedrooms = input("Minimum number of bedrooms: ")
        if min_bedrooms.strip():
            filters['min_bedrooms'] = int(min_bedrooms)

        min_bathrooms = input("Minimum number of bathrooms: ")
        if min_bathrooms.strip():
            filters['min_bathrooms'] = int(min_bathrooms)

        min_size = input("Minimum house size (sq ft): ")
        if min_size.strip():
            filters['min_size'] = float(min_size)

    except ValueError as e:
        print("\n⚠️ Invalid input detected. Skipping numeric filters.")
        return {}

    return filters

def display_matches(matches: List[Dict[str, Any]]):
    """Display matching properties in a formatted way"""
    if not matches:
        print("\n😔 No properties found matching your criteria.")
        return

    print(f"\n🎯 Found {len(matches)} matching properties!\n")
    for i, match in enumerate(matches, 1):
        print(f"\n--- Match #{i} ---")
        print(format_listing_for_display(match, include_similarity=True))
        print("\n" + "="*50)

def main():
    # Load environment variables
    load_dotenv()

    # Initialize components
    preference_model = PreferenceModel()
    llm_model = LLMModel()
    vector_store = VectorStore()

    # Print welcome message
    print_welcome()

    # Collect preferences through interactive questions
    preferences = preference_model.collect_preferences()

    # Parse preferences into structured format
    structured_preferences = preference_model.parse_preferences(preferences)

    # Generate search query
    search_query = preference_model.generate_search_query(structured_preferences)

    # Get embedding for the search query
    query_embedding = llm_model.get_embedding(search_query)

    # Collect additional numeric filters
    filters = collect_additional_filters()

    # Search vector store
    matches = vector_store.search_listings(query_embedding, n_results=5)

    # Filter results based on criteria
    filtered_matches = filter_listings_by_criteria(
        [match["metadata"] for match in matches],
        **filters
    )

    # Generate personalized descriptions
    for listing in filtered_matches:
        personalized_description = llm_model.generate_listing_description(
            listing,
            structured_preferences.__dict__
        )
        listing["personalized_description"] = personalized_description

    # Display results
    display_matches(filtered_matches)

    print("\nThank you for using HomeMatch!")

if __name__ == "__main__":
    main()