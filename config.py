import os
from dotenv import load_dotenv

load_dotenv("/opt/robin/.env")


def _clean_env(name, default=None):
	value = os.getenv(name, default)
	if value is None:
		return None
	value = str(value).strip()
	# Support accidentally quoted values copied into .env
	if len(value) >= 2 and (
		(value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
	):
		value = value[1:-1].strip()
	return value

# Configuration variables loaded from the .env file
OPENAI_API_KEY = _clean_env("OPENAI_API_KEY")
GOOGLE_API_KEY = _clean_env("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = _clean_env("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = _clean_env("OLLAMA_BASE_URL")
OPENROUTER_BASE_URL = _clean_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = _clean_env("OPENROUTER_API_KEY")
LLAMA_CPP_BASE_URL = _clean_env("LLAMA_CPP_BASE_URL")
CUSTOM_API_BASE_URL = _clean_env("CUSTOM_API_BASE_URL")
CUSTOM_API_KEY = _clean_env("CUSTOM_API_KEY")
CUSTOM_API_MODEL = _clean_env("CUSTOM_API_MODEL")

# Telemetry configuration
LANGFUSE_PUBLIC_KEY = _clean_env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _clean_env("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _clean_env("LANGFUSE_HOST", "https://cloud.langfuse.com")

PHOENIX_API_KEY = _clean_env("PHOENIX_API_KEY")
PHOENIX_ENDPOINT = _clean_env("PHOENIX_ENDPOINT", "https://app.phoenix.arize.com/v1/traces")

PRIMARY_LLM_PROVIDER = _clean_env("PRIMARY_LLM_PROVIDER", "openrouter")
FALLBACK_LLM_PROVIDER = _clean_env("FALLBACK_LLM_PROVIDER", "agy")

TELEMETRY_BATCH_SIZE = int(_clean_env("TELEMETRY_BATCH_SIZE", "50"))
TELEMETRY_FLUSH_INTERVAL = float(_clean_env("TELEMETRY_FLUSH_INTERVAL", "5.0"))
TELEMETRY_MAX_QUEUE_SIZE = int(_clean_env("TELEMETRY_MAX_QUEUE_SIZE", "10000"))
