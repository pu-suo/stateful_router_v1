"""
Configuration settings for the Enhanced Stateful Router system.

This file contains all configurable parameters for the system including:
- Model configurations
- API settings
- Verification thresholds
- Caching parameters
- Logging settings
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Central configuration management for the Stateful Router system."""
    
    # Model Configuration
    STRATEGIC_PLANNER_MODEL: str = os.getenv("STRATEGIC_PLANNER_MODEL", "gpt-4o-mini") # "gpt-4o-mini"
    TACTICAL_ROUTER_MODEL: str = os.getenv("TACTICAL_ROUTER_MODEL", "gpt-4o-mini")
    TOOL_PARSER_MODEL: str = os.getenv("TOOL_PARSER_MODEL", "gpt-3.5-turbo")
    
    # Model Temperature Settings
    PLANNER_TEMPERATURE: float = 0.2  # Low for consistency in planning
    ROUTER_TEMPERATURE: float = 0.3   # Slightly higher for flexibility
    PARSER_TEMPERATURE: float = 0.1   # Very low for extraction accuracy
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # API Endpoints (for custom deployments)
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", None)
    ANTHROPIC_API_BASE: Optional[str] = os.getenv("ANTHROPIC_API_BASE", None)
    
    # Verification Settings
    VERIFICATION_THRESHOLD: float = 0.8  # Minimum score to pass verification
    CONFIDENCE_THRESHOLD: float = 0.7    # Minimum confidence for final answer
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # Seconds between retries
    
    # Execution Limits
    MAX_STEPS_PER_PROBLEM: int = 20  # Maximum steps before forcing termination
    MAX_PLAN_ATTEMPTS: int = 3       # Maximum replanning attempts
    
    # Caching Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    CACHE_MAX_SIZE: int = 1000  # Maximum number of cached items
    
    # Parsing Configuration
    PARSE_ATTEMPTS: int = 3  # Number of attempts for probabilistic parsing
    PARSE_AGREEMENT_THRESHOLD: float = 0.66  # Minimum agreement for high confidence
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "stateful_router.log")
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / ".cache"
    
    # Strategy Definitions
    AVAILABLE_STRATEGIES: list = [
        "ARITHMETIC_SEQUENCE",
        "LOGICAL_DEDUCTION",
        "INFORMATION_EXTRACTION",
        "COMPARISON",
        "ALGEBRAIC",
        "HYBRID"
    ]
    
    # Tool Timeout Settings (in seconds)
    TOOL_TIMEOUT: Dict[str, float] = {
        "tool_add": 0.1,
        "tool_subtract": 0.1,
        "tool_multiply": 0.1,
        "tool_divide": 0.1,
        "tool_parse": 2.0,
        "tool_compare": 2.0,
        "tool_search": 5.0,
    }
    
    # Confidence Score Weights
    CONFIDENCE_WEIGHTS: Dict[str, float] = {
        "strategic_planning": 0.2,
        "tactical_execution": 0.3,
        "tool_reliability": 0.3,
        "verification_score": 0.2,
    }
    
    # Embedding Model Settings (for similarity checks)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Parallel Execution Settings
    ENABLE_PARALLEL_EXECUTION: bool = True
    MAX_PARALLEL_TOOLS: int = 3
    
    # Debug Settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    SAVE_EXECUTION_TRACES: bool = True
    TRACE_DIR: Path = BASE_DIR / "traces"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required settings are properly configured."""
        errors = []
        
        # Check API keys
        if not cls.OPENAI_API_KEY and cls.STRATEGIC_PLANNER_MODEL.startswith("gpt"):
            errors.append("OPENAI_API_KEY is required for GPT models")
        
        if not cls.ANTHROPIC_API_KEY and cls.STRATEGIC_PLANNER_MODEL.startswith("claude"):
            errors.append("ANTHROPIC_API_KEY is required for Claude models")
        
        # Check thresholds
        if not (0.0 <= cls.VERIFICATION_THRESHOLD <= 1.0):
            errors.append("VERIFICATION_THRESHOLD must be between 0 and 1")
        
        if not (0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        # Create directories if they don't exist
        for dir_path in [cls.DATA_DIR, cls.LOGS_DIR, cls.CACHE_DIR, cls.TRACE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export settings as a dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }
    
    @classmethod
    def update_from_dict(cls, updates: Dict[str, Any]) -> None:
        """Update settings from a dictionary."""
        for key, value in updates.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


# Initialize and validate settings on import
settings = Settings()
if not settings.validate():
    import warnings
    warnings.warn("Configuration validation failed. Please check your settings.")
