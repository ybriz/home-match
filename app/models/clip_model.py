import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Dict, Any
import numpy as np

class CLIPSearchModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_embedding(self, image: Union[str, Image.Image]) -> List[float]:
        """
        Get embedding for an image using CLIP
        """
        if isinstance(image, str):
            image = Image.open(image)

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        image_features = self.model.get_image_features(**inputs.to(self.device))

        # Normalize the features
        image_embedding = image_features.detach().cpu().numpy()[0]
        image_embedding = image_embedding / np.linalg.norm(image_embedding)

        return image_embedding.tolist()

    def  get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text description using CLIP
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs.to(self.device))

        # Normalize the features
        text_embedding = text_features.detach().cpu().numpy()[0]
        text_embedding = text_embedding / np.linalg.norm(text_embedding)

        return text_embedding.tolist()

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def search_similar_properties(
        self,
        query: Union[str, Image.Image],
        property_embeddings: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar properties using either text or image query
        """
        if isinstance(query, str):
            query_embedding = self.get_text_embedding(query)
        else:
            query_embedding = self.get_image_embedding(query)

        similarities = []
        for prop in property_embeddings:
            similarity = self.compute_similarity(query_embedding, prop["image_embedding"])
            similarities.append((similarity, prop))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-k results with their similarity scores
        results = []
        for similarity, prop in similarities[:top_k]:
            result = prop.copy()
            result["similarity_score"] = similarity
            results.append(result)

        return results