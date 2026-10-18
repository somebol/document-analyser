from typing import Any, Protocol


class GenerationProvider(Protocol):

    # Generate a JSON response matching the requested schema.
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        ...