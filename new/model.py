import base64
from loguru import logger
from typing import Any, Optional, List
from openai import OpenAI


class VisionAPIWrapper:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 300, temperature: float = 0.7):
        """Initializes the Vision API Wrapper for interaction with OpenAI models."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encodes an image to Base64."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            logger.debug("Image successfully encoded to Base64.")
            return encoded_string
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    def create_messages(self, task: str, base64_image: Optional[str] = None) -> List[dict]:
        """Creates the structured message payload for the API."""
        messages = [{"role": "system", "content": task}]
        user_message = {"role": "user", "content": [{"type": "text", "text": task}]}

        if base64_image:
            image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            user_message["content"].append(image_message)

        messages.append(user_message)
        logger.debug("Structured messages created for API call.")
        return messages

    def run(self, task: str, img: Optional[str] = None) -> Any:
        """Sends a request to the OpenAI API for processing a task with an optional image."""
        base64_image = self.encode_image(img) if img else None
        messages = self.create_messages(task, base64_image)

        logger.info(f"Sending request to OpenAI with task: '{task}' and image: {img}")

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature
            )
            logger.info("Received response successfully.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"An error occurred while making the API request: {e}")
            raise

    def __str__(self):
        """String representation of the Vision API Wrapper instance."""
        return f"VisionAPIWrapper(model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature})"

    def __repr__(self):
        """Detailed representation for debugging."""
        return f"<VisionAPIWrapper model={self.model} max_tokens={self.max_tokens} temperature={self.temperature}>"
