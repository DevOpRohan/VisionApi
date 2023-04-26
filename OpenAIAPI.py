import asyncio
import openai

class OpenAIAPI:
    def __init__(self, api_key, retries=3):
        self.api_key = api_key
        self.retries = retries
        openai.api_key = self.api_key

    async def chat_completion(self, model, messages, temperature, max_tokens):
        for attempt in range(self.retries):
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion
            except Exception as e:
                print(f"Error in OpenAI API call (attempt {attempt + 1}): {e}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e