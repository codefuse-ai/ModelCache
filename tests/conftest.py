"""
Shared pytest fixtures and configuration for modelcache tests.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Dict, Any
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Provide a mock configuration dictionary for testing.
    
    Returns:
        Dict[str, Any]: Mock configuration
    """
    return {
        "cache_dir": "/tmp/test_cache",
        "max_cache_size": 1000,
        "ttl": 3600,
        "embedding_model": "test-model",
        "similarity_threshold": 0.8,
        "vector_dimension": 768,
        "batch_size": 32,
        "database": {
            "type": "memory",
            "host": "localhost",
            "port": 6379,
            "password": None
        }
    }


@pytest.fixture
def mock_embedding():
    """
    Mock embedding object for testing.
    
    Returns:
        MagicMock: Mock embedding with common methods
    """
    mock = MagicMock()
    mock.embed.return_value = [0.1] * 768  # Default 768-dim embedding
    mock.embed_batch.return_value = [[0.1] * 768] * 10
    mock.dimension = 768
    mock.model_name = "test-embedding-model"
    return mock


@pytest.fixture
def mock_cache_manager():
    """
    Mock cache manager for testing.
    
    Returns:
        MagicMock: Mock cache manager with common methods
    """
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.clear.return_value = True
    mock.size.return_value = 0
    return mock


@pytest.fixture
def sample_vector_data():
    """
    Sample vector data for testing vector operations.
    
    Returns:
        Dict[str, Any]: Sample vector data
    """
    return {
        "id": "test_vector_001",
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5] * 153 + [0.6, 0.7, 0.8],  # 768 dimensions
        "metadata": {
            "source": "test",
            "timestamp": 1234567890,
            "model": "test-model"
        }
    }


@pytest.fixture
def mock_redis_client():
    """
    Mock Redis client for testing Redis-based operations.
    
    Returns:
        MagicMock: Mock Redis client
    """
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = 0
    mock.expire.return_value = True
    mock.ttl.return_value = -2
    return mock


@pytest.fixture
def mock_milvus_client():
    """
    Mock Milvus client for testing vector database operations.
    
    Returns:
        MagicMock: Mock Milvus client
    """
    mock = MagicMock()
    mock.create_collection.return_value = True
    mock.insert.return_value = MagicMock(primary_keys=[1, 2, 3])
    mock.search.return_value = [[]]
    mock.query.return_value = []
    mock.delete.return_value = MagicMock(delete_count=1)
    return mock


@pytest.fixture(autouse=True)
def reset_environment():
    """
    Reset environment variables before each test.
    """
    # Store original env vars
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["MODELCACHE_ENV"] = "test"
    os.environ["MODELCACHE_LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_text_data():
    """
    Sample text data for testing text processing.
    
    Returns:
        List[str]: List of sample texts
    """
    return [
        "This is a test sentence for modelcache.",
        "Machine learning models need efficient caching.",
        "Vector embeddings help with semantic search.",
        "Testing is important for code quality.",
        "PyTest makes testing in Python easier."
    ]


@pytest.fixture
def mock_http_response():
    """
    Mock HTTP response for testing API calls.
    
    Returns:
        MagicMock: Mock response object
    """
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"status": "success", "data": {}}
    mock.text = '{"status": "success", "data": {}}'
    mock.headers = {"Content-Type": "application/json"}
    return mock


# Pytest configuration hooks
def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Add custom markers description
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test location.
    """
    for item in items:
        # Auto-mark tests based on their location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)