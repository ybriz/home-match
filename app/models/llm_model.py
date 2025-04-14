from openai import OpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

class LLMModel:
    def __init__(self):
        self.client = client
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4-turbo-preview"

    def generate_listing_description(self, listing_data: Dict[str, Any], buyer_preferences: Dict[str, Any]) -> str:
        """
        Generate a personalized listing description based on buyer preferences
        """
        prompt = f"""
        Create a personalized real estate listing description for a potential buyer.
        
        Property Details:
        {listing_data}
        
        Buyer Preferences:
        {buyer_preferences}
        
        Generate a compelling description that highlights aspects of the property that match the buyer's preferences
        while maintaining factual accuracy. Focus on creating an engaging narrative that resonates with the buyer's
        specific interests and needs.
        """
        
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a skilled real estate agent who excels at creating personalized property descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI's embedding model
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        
        return response.data[0].embedding

    def parse_buyer_preferences(self, preferences: List[str]) -> Dict[str, Any]:
        """
        Parse buyer preferences from natural language to structured format
        """
        preferences_text = "\n".join([f"- {pref}" for pref in preferences])
        
        prompt = f"""
        Analyze these buyer preferences and extract key requirements:

        {preferences_text}

        You MUST return a valid Python dictionary in EXACTLY this format (do not include any other text):
        {{
            "property_features": {{"size": int, "bedrooms": int, "bathrooms": int}},
            "location_preferences": ["pref1", "pref2"],
            "amenities": ["amenity1", "amenity2"],
            "transportation": ["transport1", "transport2"],
            "lifestyle_factors": ["factor1", "factor2"],
            "budget_range": {{"min": int, "max": int}},
            "must_have_features": ["feature1", "feature2"],
            "nice_to_have_features": ["feature1", "feature2"]
        }}

        Rules:
        1. Use ONLY string, integer, or list values
        2. NO floating point numbers
        3. NO special characters in strings
        4. NO line breaks in strings
        5. Lists can be empty but must exist
        6. Dictionary keys must match exactly
        """

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a precise data generator that outputs only valid Python dictionaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up the response
            if "```" in result:
                # Extract content between triple backticks
                result = result.split("```")[1]
                if result.startswith("python"):
                    result = result[6:].strip()
            result = result.strip()
            
            # Ensure it's a dictionary
            if not (result.startswith("{") and result.endswith("}")):
                raise ValueError("Response must be a dictionary")
            
            # Try to evaluate the string as a Python expression
            structured_data = eval(result)
            
            # Validate the structure
            required_keys = {
                "property_features", "location_preferences", "amenities",
                "transportation", "lifestyle_factors", "budget_range",
                "must_have_features", "nice_to_have_features"
            }
            
            if not isinstance(structured_data, dict):
                raise ValueError("Result must be a dictionary")
            
            if set(structured_data.keys()) != required_keys:
                raise ValueError(f"Missing or extra keys in response")
            
            if not isinstance(structured_data["budget_range"], dict):
                raise ValueError("budget_range must be a dictionary")
            
            if not all(isinstance(v, list) for k, v in structured_data.items() 
                      if k not in ["property_features", "budget_range"]):
                raise ValueError("All values except property_features and budget_range must be lists")
            
            return structured_data
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Return a safe default structure
            return {
                "property_features": {"size": 0, "bedrooms": 0, "bathrooms": 0},
                "location_preferences": [],
                "amenities": [],
                "transportation": [],
                "lifestyle_factors": [],
                "budget_range": {"min": 0, "max": 999999999},
                "must_have_features": [],
                "nice_to_have_features": []
            }

    def generate_sample_listings(self, num_listings: int = 10) -> List[Dict[str, Any]]:
        """
        Generate sample real estate listings using the LLM
        """
        prompt = f"""
        Generate exactly {num_listings} real estate listings in a Python list.
        
        IMPORTANT: Return ONLY a Python list containing dictionaries. No other text or explanation.
        Each dictionary MUST have these exact keys and value types:
        
        {{
            'neighborhood': (string),
            'price': (integer, no commas or currency symbols),
            'bedrooms': (integer),
            'bathrooms': (integer),
            'house_size': (integer, square feet),
            'description': (string),
            'neighborhood_description': (string)
        }}

        Example of EXACT format:
        [{{
            'neighborhood': 'Green Valley',
            'price': 450000,
            'bedrooms': 3,
            'bathrooms': 2,
            'house_size': 2000,
            'description': 'Beautiful modern home with updated kitchen',
            'neighborhood_description': 'Quiet suburban area with parks'
        }}, {{
            'neighborhood': 'Downtown',
            'price': 750000,
            'bedrooms': 2,
            'bathrooms': 2,
            'house_size': 1500,
            'description': 'Luxury condo with city views',
            'neighborhood_description': 'Vibrant urban center with restaurants'
        }}]

        Ensure:
        - Price range varies from 300000 to 2000000 (integers only)
        - Mix of property types (houses, condos, townhomes)
        - Diverse neighborhoods (urban, suburban, rural)
        - Varied amenities and features
        - NO special characters in strings
        - NO line breaks in strings
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a precise data generator that outputs only valid Python lists containing real estate listings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove any markdown code block syntax
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("python"):
                    result = result[6:].strip()
            
            # Validate and clean the response
            result = result.strip()
            if not (result.startswith("[") and result.endswith("]")):
                raise ValueError("Response must be a list")
            
            # Try to evaluate the string as a Python expression
            listings = eval(result)
            
            # Validate the structure
            if not isinstance(listings, list):
                raise ValueError("Result must be a list")
            
            required_keys = {'neighborhood', 'price', 'bedrooms', 'bathrooms', 
                           'house_size', 'description', 'neighborhood_description'}
            
            for listing in listings:
                if not isinstance(listing, dict):
                    raise ValueError("Each item must be a dictionary")
                if set(listing.keys()) != required_keys:
                    raise ValueError(f"Missing or extra keys in listing: {set(listing.keys())}")
                if not isinstance(listing['price'], int):
                    raise ValueError("Price must be an integer")
                if not isinstance(listing['bedrooms'], int):
                    raise ValueError("Bedrooms must be an integer")
                if not isinstance(listing['bathrooms'], int):
                    raise ValueError("Bathrooms must be an integer")
                if not isinstance(listing['house_size'], int):
                    raise ValueError("House size must be an integer")
            
            return listings
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Return a list with one default listing
            return [{
                'neighborhood': 'Sample Neighborhood',
                'price': 500000,
                'bedrooms': 3,
                'bathrooms': 2,
                'house_size': 2000,
                'description': 'A lovely sample home in a great location',
                'neighborhood_description': 'Peaceful neighborhood with easy access to amenities'
            }] 