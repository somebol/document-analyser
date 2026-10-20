from typing import Any

from ollama import Client

from document_analyser.config import get_settings


class OllamaGenerationProvider:
    # Configure generation from shared settings while allowing explicit overrides.
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.model = model if model is not None else settings.generation_model
        self.client = Client(
            host=host if host is not None else settings.ollama_host,
            timeout=settings.generation_timeout,
        )

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
                "temperature": self.settings.generation_temperature,
                "num_ctx": self.settings.generation_context_tokens,
                "num_predict": self.settings.generation_max_tokens,
            },
        )

        if response.done_reason == "length":
            raise RuntimeError("The answer exceeded the output budget.")

        content = response.message.content

        if not content:
            raise RuntimeError("The model returned an empty answer.")

        return content
