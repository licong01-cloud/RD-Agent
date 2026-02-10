"""
LLM Configuration Management Module for RD-Agent.

This module provides API endpoints for managing LLM configurations,
including providers, models, and stage mappings.
"""

from .routes import router

__all__ = ["router"]
