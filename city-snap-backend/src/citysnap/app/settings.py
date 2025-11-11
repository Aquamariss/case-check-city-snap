"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional



"""@dataclass(frozen=True)"""

class AppSettings:
    """Minimal settings container for application-wide configuration."""

    """open_api_key: Optional[str] """
    """giga_chat_api_key: Optional[str] """
    def __init__(self) -> None:
        self.open_api_key: Optional[str] = os.getenv("OPEN_API_KEY")
        self.giga_chat_api_key: Optional[str] = os.getenv("GIGA_CHAT_API_KEY")

_settings_instance: Optional[AppSettings] = None 

"""@lru_cache(maxsize=1)"""

def get_app_settings() -> AppSettings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance



"""def get_app_settings() -> AppSettings:
    Return cached application settings parsed from the environment.
    return AppSettings(
        open_api_key=_coerce_env(os.getenv("OPEN_API_KEY")),
        giga_chat_api_key=_coerce_env(os.getenv("GIGA_CHAT_API_KEY")),
    )"""


def _coerce_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def reset_settings_cache() -> None:
    """Clear cached settings; intended for use in test suites."""
    get_app_settings.cache_clear()


__all__ = ["AppSettings", "get_app_settings", "reset_settings_cache"]
