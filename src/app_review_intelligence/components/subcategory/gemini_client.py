"""Gemini API transport seam for subcategory discovery.

Provides single-request transport and transient error checking.
Contains no retry loops, response parsing, or validation logic.
"""

import os

DEFAULT_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL_ID = "gemini-3.6-flash"


def check_api_key(env_var_name=DEFAULT_API_KEY_ENV_VAR):
    """
    Checks that the specified environment variable exists and is not empty.

    Args:
        env_var_name: Name of the environment variable (default 'GEMINI_API_KEY').

    Raises:
        ValueError: If the environment variable is not set or empty.
            Names the variable in the message; never logs or prints any value.
    """
    value = os.environ.get(env_var_name)
    if not value or not value.strip():
        raise ValueError(f"Environment variable '{env_var_name}' is not set or empty.")


def is_transient_error(exception):
    """
    Determines whether an exception represents a transient API error.

    Transient errors include rate limits (429), server errors (5xx),
    timeouts, and temporary connection failures.

    Args:
        exception: The Exception to inspect.

    Returns:
        bool: True if the exception is transient and safe to retry with backoff.
    """
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return True

    code = getattr(exception, "code", None) or getattr(exception, "status_code", None)
    if isinstance(code, int):
        if code == 429 or 500 <= code < 600:
            return True

    try:
        from google.genai.errors import APIError, ServerError

        if isinstance(exception, ServerError):
            return True
        if isinstance(exception, APIError):
            if exception.code == 429 or (exception.code and 500 <= exception.code < 600):
                return True
    except ImportError:
        pass

    msg = str(exception).lower()
    transient_indicators = (
        "429",
        "rate limit",
        "quota",
        "resource exhausted",
        "503",
        "500",
        "502",
        "504",
        "server error",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    )
    return any(indicator in msg for indicator in transient_indicators)


def generate_content(
    system_prompt,
    user_prompt,
    temperature=0.0,
    model_id=None,
    seed=None,
):
    """
    Performs a single Gemini generate_content request and returns the response text.

    Args:
        system_prompt: System instruction text.
        user_prompt: User instruction and batch JSON string.
        temperature: Generation temperature (default 0.0).
        model_id: Model identifier (default 'gemini-3.6-flash').
        seed: Optional integer random seed for reproducibility.

    Returns:
        str: Raw response text from the model.

    Raises:
        ValueError: If the API key environment variable is missing or empty.
        Exception: On network, client, or server failure.
    """
    env_var_name = os.environ.get("GEMINI_API_KEY_ENV_VAR", DEFAULT_API_KEY_ENV_VAR)
    check_api_key(env_var_name)
    api_key = os.environ[env_var_name]

    import google.genai as genai
    from google.genai import types

    target_model = model_id or DEFAULT_MODEL_ID

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        temperature=temperature,
        seed=seed,
    )
    response = client.models.generate_content(
        model=target_model,
        contents=user_prompt,
        config=config,
    )
    return response.text
