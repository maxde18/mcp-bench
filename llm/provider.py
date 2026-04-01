"""LLM Provider Module.

This module handles all interactions with OpenRouter and other OpenAI-compatible
LLM services, providing a unified interface for model communication with retry
logic and error handling.

Classes:
    LLMProvider: Universal LLM provider for OpenAI-compatible endpoints
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
import json_repair

logger = logging.getLogger(__name__)

# Reasoning models that do not accept a custom temperature value.
# Passing temperature to these models causes an API error.
# Includes both short names and OpenRouter-prefixed names.
MODELS_WITHOUT_TEMPERATURE: Set[str] = {
    "o1-preview", "o1-mini", "o4-mini", "o3-mini", "o3",
    "openai/o1-preview", "openai/o1-mini", "openai/o4-mini", "openai/o3-mini", "openai/o3",
}


class LLMProvider:
    """LLM provider for OpenAI-compatible endpoints (OpenRouter etc.).

    Provides a unified interface for interacting with LLM APIs with built-in
    retry logic, error handling, and token management.

    Attributes:
        client: The async OpenAI-compatible client instance
        deployment_name: Model identifier (e.g. ``"qwen/qwen3-32b"``)

    Example:
        >>> from openai import AsyncOpenAI
        >>> client = AsyncOpenAI(api_key=..., base_url="https://openrouter.ai/api/v1")
        >>> provider = LLMProvider(client, "qwen/qwen3-32b")
        >>> response = await provider.get_completion("You are helpful", "Hello", 100)
    """

    def __init__(self, client: Any, deployment_name: str) -> None:
        """Initialize the LLM provider.

        Args:
            client: Async OpenAI-compatible client instance
            deployment_name: Model identifier to use for API calls
        """
        self.client = client
        self.deployment_name: str = deployment_name

    def _is_token_limit_error(self, error_message: str) -> bool:
        """Check if the error is related to token limits."""
        error_lower = str(error_message).lower()
        token_limit_indicators = [
            "maximum context length",
            "context length",
            "token limit",
            "too many tokens",
            "exceeds maximum",
            "requested too many tokens"
        ]
        return any(indicator in error_lower for indicator in token_limit_indicators)

    def _is_content_filter_error(self, error_message: str) -> bool:
        """Check if the error is related to content filtering."""
        error_lower = str(error_message).lower()
        content_filter_indicators = [
            "content management policy",
            "content filtering policies",
            "content_filter",
            "jailbreak",
            "responsibleaipolicyviolation"
        ]
        return any(indicator in error_lower for indicator in content_filter_indicators)

    def _extract_requested_tokens(self, error_message: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract requested and max tokens from a token-limit error message."""
        import re

        pattern = r"you requested (\d+) tokens.*maximum context length is (\d+) tokens"
        match = re.search(pattern, str(error_message), re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))

        pattern2 = r"(\d+) tokens.*in the messages.*(\d+) in the completion"
        match2 = re.search(pattern2, str(error_message), re.IGNORECASE)
        if match2:
            return int(match2.group(1)) + int(match2.group(2)), None

        return None, None

    async def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        return_usage: bool = False,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a completion from the LLM with retry mechanism.

        Args:
            system_prompt: System message to set context
            user_prompt: User's input prompt
            max_tokens: Maximum tokens for the response
            return_usage: If True, returns tuple of (content, usage_dict)
            temperature: Sampling temperature (0.0–2.0). Ignored for reasoning
                models (o-series) which do not accept this parameter.
            response_format: Structured output schema (e.g. OpenRouter
                ``json_schema`` format). Passed verbatim to the API when set.

        Returns:
            The LLM's response as a string, or tuple of (content, usage_dict)
            if return_usage=True.

        Raises:
            Exception: If all retry attempts fail
        """
        params = {
            "model": self.deployment_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            if self.deployment_name in MODELS_WITHOUT_TEMPERATURE:
                logger.warning(
                    f"Model '{self.deployment_name}' does not support custom temperature. "
                    "Ignoring temperature parameter."
                )
            else:
                params["temperature"] = temperature

        if response_format is not None:
            params["response_format"] = response_format

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Generating completion using {self.deployment_name} (attempt {attempt + 1}/{max_attempts}, max_tokens: {max_tokens})")

                response = await self.client.chat.completions.create(**params)
                content = response.choices[0].message.content

                if content is None or content.strip() == "":
                    raise ValueError("Empty content received from LLM")

                if attempt > 0:
                    logger.info(f"Success on attempt {attempt + 1} for {self.deployment_name}")

                if return_usage and hasattr(response, 'usage') and response.usage:
                    usage_dict = {
                        'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                        'total_tokens': getattr(response.usage, 'total_tokens', 0)
                    }
                    return content.strip(), usage_dict

                return content.strip()

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {self.deployment_name}: {e}")

                if self._is_content_filter_error(error_msg):
                    logger.info(f"Content filter error detected for {self.deployment_name}, failing fast")
                    raise e

                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

        raise Exception("Unexpected completion flow")

    async def get_completion_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        max_tokens: int,
        tool_choice: str = "none",
        return_usage: bool = False,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a completion with MCP tools passed as native OpenAI tool definitions.

        Passes ``tools`` in the structured ``tools`` field of the API request
        rather than embedding tool descriptions in the prompt text.  This is
        the correct input format for fine-tuned tool-calling models.

        When ``tool_choice="none"`` (default) the model is aware of the tools
        but must respond with plain text — suitable for planning agents that
        output a DAG as JSON.  Set ``tool_choice="auto"`` to let the model
        decide whether to call a tool.

        Args:
            system_prompt: System message to set context.
            user_prompt:   User's input prompt (task description etc.).
            tools:         OpenAI-format tool list, e.g. from
                           ``MCPConnector.format_tools_for_api(all_tools)``.
            max_tokens:    Maximum tokens for the response.
            tool_choice:   ``"none"`` | ``"auto"`` | ``{"type": "function",
                           "function": {"name": "..."}}`` — passed verbatim to
                           the API.  Defaults to ``"none"``.
            return_usage:  If True, returns ``(content, usage_dict)`` instead
                           of just the content string.
            temperature:   Sampling temperature (0.0–2.0).  Ignored for
                           reasoning models (o-series).
            response_format: Structured output schema (e.g. OpenRouter
                           ``json_schema`` format). Passed verbatim to the
                           API when set.

        Returns:
            The model's text response (or tool-call JSON when tool_choice
            forces a specific function), or a tuple ``(content, usage_dict)``
            when ``return_usage=True``.
        """
        params: Dict[str, Any] = {
            "model": self.deployment_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "tools":       tools,
            "tool_choice": tool_choice,
            "max_tokens":  max_tokens,
        }

        if temperature is not None:
            if self.deployment_name in MODELS_WITHOUT_TEMPERATURE:
                logger.warning(
                    f"Model '{self.deployment_name}' does not support custom temperature. "
                    "Ignoring temperature parameter."
                )
            else:
                params["temperature"] = temperature

        if response_format is not None:
            params["response_format"] = response_format

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Generating completion (native tools) using {self.deployment_name} "
                    f"(attempt {attempt + 1}/{max_attempts}, tool_choice={tool_choice!r})"
                )

                response = await self.client.chat.completions.create(**params)
                message = response.choices[0].message

                if message.tool_calls:
                    content = message.tool_calls[0].function.arguments
                else:
                    content = message.content

                if content is None or (isinstance(content, str) and content.strip() == ""):
                    raise ValueError("Empty content received from LLM")

                if isinstance(content, str):
                    content = content.strip()

                if return_usage and hasattr(response, "usage") and response.usage:
                    usage_dict = {
                        "prompt_tokens":     getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens":      getattr(response.usage, "total_tokens", 0),
                    }
                    return content, usage_dict

                return content

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for {self.deployment_name}: {e}"
                )
                if self._is_content_filter_error(error_msg):
                    raise e
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise e

        raise Exception("Unexpected completion flow")

    def clean_and_parse_json(self, raw_json: str) -> Any:
        """Clean and parse JSON response with enhanced error handling."""
        try:
            if '```json' in raw_json:
                raw_json = raw_json.split('```json')[1].split('```')[0].strip()
            elif '```' in raw_json:
                parts = raw_json.split('```')
                if len(parts) >= 2:
                    raw_json = parts[1].strip()

            raw_json = raw_json.strip()
            if not raw_json.startswith('{') and not raw_json.startswith('['):
                first_brace = raw_json.find('{')
                first_bracket = raw_json.find('[')

                if first_brace == -1 and first_bracket == -1:
                    logger.error(f"No JSON object or array found in the raw response: {raw_json}")
                    raise ValueError(f"No JSON object or array found in LLM response: {raw_json[:500]}")

                start_idx = -1
                if first_brace != -1 and first_bracket != -1:
                    start_idx = min(first_brace, first_bracket)
                elif first_brace != -1:
                    start_idx = first_brace
                else:
                    start_idx = first_bracket

                if start_idx != -1:
                    raw_json = raw_json[start_idx:]

            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                return json_repair.loads(raw_json)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response: {raw_json[:500]}...")
            raise ValueError(f"Failed to parse JSON from LLM response: {e}. Raw response: {raw_json[:500]}")
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}")
            raise ValueError(f"Unexpected error parsing JSON from LLM response: {e}. Raw response: {raw_json[:500] if 'raw_json' in locals() else 'N/A'}")
