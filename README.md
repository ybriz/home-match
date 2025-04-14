# HomeMatch

HomeMatch is an AI-powered real estate matching application that uses natural language processing to understand your preferences and find your perfect home. The application leverages Large Language Models (LLMs) and vector databases to transform your requirements into personalized property recommendations.

## Features

- **Natural Language Preference Collection**: Share your preferences in plain English - no need to fill out complex forms
- **Smart Preference Analysis**: Uses AI to understand both explicit and implicit requirements
- **Personalized Property Matching**: Finds properties that match your lifestyle and specific needs
- **Customized Descriptions**: Generates property descriptions tailored to your interests
- **Flexible Filtering**: Combine AI-powered matching with traditional filters (price, size, etc.)
- **REST API**: Access all functionality programmatically through a REST API
- **Sample Generation**: Generate realistic sample listings for testing and demonstration

## Prerequisites

- Python 3.11
- pip (Python package installer)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/home-match.git
cd home-match
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables by creating a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Using the Console Application

Run the console application:
```bash
python app/console_app.py
```

The application will guide you through the following steps:

1. **Share Your Preferences**: Answer questions about your ideal home, such as:
   - Desired house size
   - Most important features
   - Preferred amenities
   - Transportation requirements
   - Neighborhood preferences

2. **Set Specific Filters** (optional):
   - Minimum/maximum price
   - Minimum number of bedrooms
   - Minimum number of bathrooms
   - Minimum house size

3. **Review Matches**: The application will:
   - Find properties matching your preferences
   - Generate personalized descriptions highlighting relevant features
   - Display detailed property information including:
     - Price and basic features
     - Neighborhood description
     - Match score
     - Personalized property description

## Generating Sample Listings

You can generate sample listings in two ways:

1. Using the command-line script:
```bash
python app/scripts/generate_listings.py [number_of_listings]
```
Example:
```bash
python app/scripts/generate_listings.py 20  # Generates 20 sample listings
```

2. Using the REST API endpoint (see API documentation below)

The generated listings will be saved in `data/listings.json` and automatically loaded into the vector store for searching.

## REST API

HomeMatch also provides a REST API that can be used to access all functionality programmatically. To start the API server:


```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Testing API Endpoints

The project includes an `api_client.http` file for testing the API endpoints using the VSCode REST Client extension. To use it:

1. Install the "REST Client" extension in VSCode
2. Open `api_client.http`
3. Click "Send Request" above any of the requests to test them

Available test endpoints:

```http
### Health Check
GET http://localhost:8000/health

### Generate Sample Listings
POST http://localhost:8000/generate-listings
Content-Type: application/json

{
    "num_listings": 10
}

### Search Listings
POST http://localhost:8000/search-listings?min_price=400000&max_price=800000&min_bedrooms=3&min_bathrooms=2&min_size=2000&n_results=5
Content-Type: application/json

[
    "A comfortable three-bedroom house with a spacious kitchen",
    "A quiet neighborhood with good schools",
    "Modern amenities and a backyard",
    "Close to public transportation",
    "Suburban area with easy access to shops"
]

### Image Search
POST http://localhost:8000/image-search
Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryABC123
# Note: Requires sample_house.jpg in the data directory
```


