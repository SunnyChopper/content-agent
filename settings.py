import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Model Settings
## OpenAI
DEFAULT_OPENAI_SMART_MODEL = 'gpt-4o'
DEFAULT_OPENAI_CHEAP_MODEL = 'gpt-4o-mini'
DEFAULT_OPENAI_SPEED_MODEL = 'gpt-4o-mini'

## Google
DEFAULT_GOOGLE_SMART_MODEL = 'gemini-1.5-pro'
DEFAULT_GOOGLE_CHEAP_MODEL = 'gemini-1.5-flash'
DEFAULT_GOOGLE_SPEED_MODEL = 'gemini-1.5-flash'

## Anthropic
DEFAULT_ANTHROPIC_SMART_MODEL = 'claude-3-5-sonnet-20241022'
DEFAULT_ANTHROPIC_CHEAP_MODEL = 'claude-3-5-haiku-20241022'
DEFAULT_ANTHROPIC_SPEED_MODEL = 'claude-3-5-haiku-20241022'

# Misc. Settings
MAX_TOKENS = 4096
MAX_RESEARCH_LOOPS = 2