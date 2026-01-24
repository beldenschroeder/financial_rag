"""Pytest configuration and fixtures for financial-rag tests."""

import pytest


@pytest.fixture
def sample_query():
    """Fixture providing a sample financial query."""
    return "What were my total expenses in January 2024?"


@pytest.fixture
def mock_documents():
    """Fixture providing mock financial documents."""
    return [
        {
            "content": "January 2024 Expenses: $1,500 on groceries, $200 on gas",
            "metadata": {"date": "2024-01-01", "type": "expenses"},
        },
        {
            "content": "January 2024 Income: $5,000 from salary",
            "metadata": {"date": "2024-01-01", "type": "income"},
        },
    ]
