from typing import Any

from ollama import Client


class OllamaGenerationProvider:

    # Configure the local generative model independently of embeddings.
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.client = Client(host=host, timeout=300.0)

    # Request schema-constrained JSON from Ollama.
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=schema,
            think=False,
            options={
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 1500,
            },
        )

        if response.done_reason == "length":
            raise RuntimeError("The answer exceeded the output budget.")

        content = response.message.content

        if not content:
            raise RuntimeError("The model returned an empty answer.")

        return content