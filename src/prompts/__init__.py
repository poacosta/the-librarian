"""
Prompts package for The Librarian.

This package contains all prompt templates and conversation management
for the Borges expert RAG system.

Templates:
- BORGES_EXPERT_TEMPLATE: Main expert persona prompt
- BORGES_PROMPT: LangChain PromptTemplate instance
- CONVERSATION_STARTERS: Example questions for the interface
"""

from .templates import (
    BORGES_EXPERT_TEMPLATE,
    BORGES_PROMPT,
    CONVERSATION_STARTERS
)

__all__ = [
    "BORGES_EXPERT_TEMPLATE",
    "BORGES_PROMPT",
    "CONVERSATION_STARTERS",
]


def validate_prompt_template(template_str: str) -> bool:
    """
    Validate that a prompt template contains required variables.

    Args:
        template_str: The prompt template string to validate

    Returns:
        bool: True if the template contains required variables
    """
    required_vars = ["context", "question"]
    return all(f"{{{var}}}" in template_str for var in required_vars)


__all__.append("validate_prompt_template")
