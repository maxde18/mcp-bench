"""LLM Factory Module.

This module handles model configuration and factory creation for OpenRouter
models, providing a simple interface to list available models and create
LLMProvider instances.

Classes:
    ModelConfig: Configuration container for a specific model
    LLMFactory: Factory for creating LLM provider instances
"""

import os
from typing import Dict
from openai import AsyncOpenAI
from langchain_openrouter import ChatOpenRouter
from .provider import LLMProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelConfig:
    """Configuration for a single OpenRouter model.

    Attributes:
        name: Short identifier used as the CLI --model argument
        model_name: Full OpenRouter model ID (e.g. ``"qwen/qwen3-32b"``)

    Example:
        >>> config = ModelConfig("qwen-3-32b", "qwen/qwen3-32b")
    """

    def __init__(self, name: str, model_name: str) -> None:
        self.name: str = name
        self.model_name: str = model_name


class LLMFactory:
    """Factory for creating LLM providers for OpenRouter models.

    All models are accessed through the OpenRouter API using the
    ``OPENROUTER_API_KEY`` environment variable.

    Example:
        >>> configs = LLMFactory.get_model_configs()
        >>> provider = await LLMFactory.create_llm_provider(configs["qwen-3-32b"])
    """

    @staticmethod
    def get_model_configs() -> Dict[str, ModelConfig]:
        """Return all available OpenRouter model configurations.

        Models are only included when ``OPENROUTER_API_KEY`` is set.

        Returns:
            Dictionary mapping short model names to ModelConfig instances.
        """
        configs: Dict[str, ModelConfig] = {}

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return configs

        openrouter_models = [
            ("qwen-3-32b",                       "qwen/qwen3-32b"),
            ("qwen3-30b-a3b-instruct-2507",       "qwen/qwen3-30b-a3b-instruct-2507"),
            ("qwen3-235b-a22b-thinking-2507",     "qwen/qwen3-235b-a22b-thinking-2507"),
            ("qwen3-235b-a22b-2507",              "qwen/qwen3-235b-a22b-2507"),
            ("gpt-oss-20b",                       "openai/gpt-oss-20b"),
            ("gpt-oss-120b",                      "openai/gpt-oss-120b"),
            ("kimi-k2",                           "moonshotai/kimi-k2"),
            ("minimax-m1",                        "minimax/minimax-m1"),
            ("nova-micro-v1",                     "amazon/nova-micro-v1"),
            ("grok-3-mini",                       "x-ai/grok-3-mini"),
            ("gemini-2.5-flash-lite",             "google/gemini-2.5-flash-lite"),
            ("gpt-5-mini",                        "openai/gpt-5-mini"),
            ("gpt-5-nano",                        "openai/gpt-5-nano"),
            ("deepseek-r1-0528",                  "deepseek/deepseek-r1-0528"),
            ("deepseek-r1-0528-qwen3-8b",         "deepseek/deepseek-r1-0528-qwen3-8b"),
            ("ernie-4.5-21b-a3b",                 "baidu/ernie-4.5-21b-a3b"),
            ("glm-4.5-air",                       "z-ai/glm-4.5-air"),
            ("mistral-small-3.2-24b-instruct",    "mistralai/mistral-small-3.2-24b-instruct"),
            ("gemma-3-27b-it",                    "google/gemma-3-27b-it"),
            ("qwq-32b",                           "qwen/qwq-32b"),
            ("glm-4.5",                           "z-ai/glm-4.5"),
            ("claude-sonnet-4",                   "anthropic/claude-sonnet-4"),
            ("gemini-2.5-pro",                    "google/gemini-2.5-pro"),
            ("minimax-m2.7",                      "minimax/minimax-m2.7"),
        ]

        for name, model_name in openrouter_models:
            configs[name] = ModelConfig(name=name, model_name=model_name)

        return configs

    @staticmethod
    async def create_llm_provider(model_config: ModelConfig) -> LLMProvider:
        """Create an LLMProvider for the given model configuration.

        Args:
            model_config: Configuration for the model to create.

        Returns:
            Configured LLMProvider instance pointed at OpenRouter.
        """
        client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )
        return LLMProvider(client=client, deployment_name=model_config.model_name)

    @staticmethod
    def create_chat_model(model_config: ModelConfig, max_retries: int = 3) -> ChatOpenRouter:
        """Create a ChatOpenRouter model for the given model configuration.

        Uses the ``langchain-openrouter`` integration, which provides native
        LangGraph support, tool binding via ``bind_tools()``, and structured
        output via ``with_structured_output()`` or ``bind(response_format=...)``.

        The ``OPENROUTER_API_KEY`` environment variable is read automatically.

        Args:
            model_config: Configuration for the model to create.
            max_retries:  Number of automatic retry attempts on transient API
                          errors (default: 3).

        Returns:
            Configured ChatOpenRouter instance.
        """
        return ChatOpenRouter(
            model=model_config.model_name,
            max_retries=max_retries,
        )
