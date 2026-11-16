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


def is_quota_exhausted(exception):
    """
    True when Gemini reports an exhausted request quota (daily/plan limit).

    Retrying cannot help, so the caller stops calling the API. Short per-minute
    rate limits are not treated as exhausted quota and stay retryable.
    """
    msg = str(exception).lower()
    if "quota" not in msg and "resource_exhausted" not in msg and "resource exhausted" not in msg:
        return False
    return any(marker in msg for marker in ("perday", "per day", "check your plan", "billing"))


def is_transient_error(exception):
    """
    Determines whether an exception represents a transient API error.

    Transient errors include rate limits (429), server errors (5xx),
    timeouts, and temporary connection failures. 4xx client errors
    (other than 429) are considered permanent.

    Args:
        exception: The Exception to inspect.

    Returns:
        bool: True if the exception is transient and safe to retry with backoff.
    """
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return True

    # Stand-in or real exception attributes: status_code, code, status
    code = getattr(exception, "status_code", None)
    if code is None:
        code = getattr(exception, "code", None)
    if code is None:
        code = getattr(exception, "status", None)

    if isinstance(code, int):
        if code == 429 or 500 <= code < 600:
            return True
        if 400 <= code < 500:
            return False

    if isinstance(code, str):
        if code.isdigit():
            c = int(code)
            if c == 429 or 500 <= c < 600:
                return True
            if 400 <= c < 500:
                return False
        upper = code.upper()
        if upper in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED"):
            return True
        if upper in ("INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED", "UNAUTHENTICATED"):
            return False

    try:
        from google.genai.errors import APIError, ClientError, ServerError

        if isinstance(exception, ServerError):
            return True
        if isinstance(exception, ClientError):
            if getattr(exception, "code", None) == 429:
                return True
            return False
        if isinstance(exception, APIError):
            if exception.code == 429 or (exception.code and 500 <= exception.code < 600):
                return True
            if exception.code and 400 <= exception.code < 500:
                return False
    except ImportError:
        pass

    msg = str(exception).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg or "resource exhausted" in msg:
        return True
    if any(
        s in msg
        for s in (
            "500",
            "502",
            "503",
            "504",
            "server error",
            "timeout",
            "timed out",
            "connection reset",
            "temporarily unavailable",
        )
    ):
        return True

    return False


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


def check_model_access(model_id=None, env_var_name=None, timeout_seconds=30):
    """
    Minimal safe pre-flight check: fetches the model's metadata from Gemini.

    Sends no review text and no prompt; it only confirms that the API key,
    network and model ID work before a run starts.

    Raises:
        ValueError: If the API key environment variable is missing or empty.
        RuntimeError: If the Gemini API cannot be reached or rejects the request.
    """
    env_var_name = env_var_name or os.environ.get("GEMINI_API_KEY_ENV_VAR", DEFAULT_API_KEY_ENV_VAR)
    check_api_key(env_var_name)

    import google.genai as genai
    from google.genai import types

    target_model = model_id or DEFAULT_MODEL_ID
    try:
        client = genai.Client(
            api_key=os.environ[env_var_name],
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        client.models.get(model=target_model)
    except Exception as exc:
        # The message never contains the key; it names the model and the error type only.
        raise RuntimeError(
            f"Gemini API check failed for model '{target_model}': {type(exc).__name__}: {exc}"
        ) from exc
