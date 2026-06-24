from mistralai import Mistral
import os

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def get_mistral_client() -> Mistral:
    """
    Returns a Mistral client instance.
    """
    return client