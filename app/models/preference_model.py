from typing import List, Dict, Any
from dataclasses import dataclass
from .llm_model import LLMModel

@dataclass
class BuyerPreferences:
    property_features: Dict[str, Any]
    location_preferences: List[str]
    amenities: List[str]
    transportation: List[str]
    lifestyle_factors: List[str]
    budget_range: Dict[str, float]
    must_have_features: List[str]
    nice_to_have_features: List[str]

class PreferenceModel:
    def __init__(self):
        self.llm_model = LLMModel()
        self.default_questions = [
            "How big do you want your house to be?",
            "What are 3 most important things for you in choosing this property?",
            "Which amenities would you like?",
            "Which transportation options are important to you?",
            "How urban do you want your neighborhood to be?",
            "What is your budget range?",
            "Are there any must-have features?",
            "What would be nice to have but isn't essential?"
        ]

    def collect_preferences(self, answers: List[str] = None) -> List[str]:
        """
        Collect buyer preferences either from provided answers or interactively
        """
        if answers:
            return answers

        collected_answers = []
        print("\nPlease answer the following questions about your preferences:")
        for question in self.default_questions:
            answer = input(f"\n{question}\n> ")
            collected_answers.append(answer)
        return collected_answers

    def parse_preferences(self, preferences: List[str]) -> BuyerPreferences:
        """
        Parse natural language preferences into structured format using LLM
        """
        prompt = f"""
        Analyze these buyer preferences and extract key requirements into specific categories.

        Buyer Preferences:
        {preferences}

        Format the output as a Python dictionary with these categories:
        - property_features: dict with size, bedrooms, bathrooms, etc.
        - location_preferences: list of location requirements
        - amenities: list of desired amenities
        - transportation: list of transportation preferences
        - lifestyle_factors: list of lifestyle considerations
        - budget_range: dict with min and max if mentioned
        - must_have_features: list of essential features
        - nice_to_have_features: list of desired but non-essential features

        Ensure the output is valid Python code that can be evaluated.
        """

        # Get structured response from LLM
        structured_data = self.llm_model.parse_buyer_preferences(preferences)

        # Convert to BuyerPreferences object
        return BuyerPreferences(
            property_features=structured_data.get('property_features', {}),
            location_preferences=structured_data.get('location_preferences', []),
            amenities=structured_data.get('amenities', []),
            transportation=structured_data.get('transportation', []),
            lifestyle_factors=structured_data.get('lifestyle_factors', []),
            budget_range=structured_data.get('budget_range', {'min': 0, 'max': float('inf')}),
            must_have_features=structured_data.get('must_have_features', []),
            nice_to_have_features=structured_data.get('nice_to_have_features', [])
        )

    def generate_search_query(self, preferences: BuyerPreferences) -> str:
        """
        Generate a search query from structured preferences
        """
        # Combine all relevant preferences into a search query
        query_parts = []

        # Add property features
        if preferences.property_features:
            features = [f"{k}: {v}" for k, v in preferences.property_features.items()]
            query_parts.extend(features)

        # Add location preferences
        query_parts.extend(preferences.location_preferences)

        # Add must-have features
        query_parts.extend(preferences.must_have_features)

        # Add key amenities
        query_parts.extend(preferences.amenities[:3])  # Top 3 amenities

        # Add lifestyle factors
        query_parts.extend(preferences.lifestyle_factors)

        # Combine into a search query
        search_query = " ".join(query_parts)

        return search_query