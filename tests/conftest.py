"""Pytest configuration and fixtures."""

import pytest
import asyncio


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_tokens():
    """Sample token list for testing."""
    return list(range(100))


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "What is artificial intelligence?"
