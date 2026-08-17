"""
Test to verify the ExpiredTokenException retry logic works correctly.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from botocore.exceptions import ClientError


async def test_invoke_with_retry_success():
    """Test that invoke_with_retry works on first attempt"""
    from app.core.llm.chat_models import invoke_with_retry
    
    # Mock model that succeeds
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value="Success response")
    
    result = await invoke_with_retry(mock_model, [{"role": "user", "content": "test"}], efficient=True)
    
    assert result == "Success response"
    assert mock_model.ainvoke.call_count == 1
    print("✅ Test 1 passed: invoke_with_retry succeeds on first attempt")


async def test_invoke_with_retry_token_expiry_then_success():
    """Test that invoke_with_retry retries on ExpiredTokenException"""
    from app.core.llm.chat_models import invoke_with_retry, get_model
    
    # Create a mock that fails once then succeeds
    mock_model = AsyncMock()
    
    # First call raises ExpiredTokenException
    expired_error = ClientError(
        {'Error': {'Code': 'ExpiredTokenException', 'Message': 'Token expired'}},
        'Converse'
    )
    
    # Set up the mock to fail once, then succeed
    mock_model.ainvoke = AsyncMock(side_effect=[expired_error, "Success after retry"])
    
    # Mock get_model to return our mock model
    with patch('app.core.llm.chat_models.get_model', return_value=mock_model):
        result = await invoke_with_retry(mock_model, [{"role": "user", "content": "test"}], efficient=True, max_retries=1)
    
    assert result == "Success after retry"
    assert mock_model.ainvoke.call_count == 2
    print("✅ Test 2 passed: invoke_with_retry retries after ExpiredTokenException")


async def test_invoke_with_retry_max_retries_exceeded():
    """Test that invoke_with_retry raises after max retries"""
    from app.core.llm.chat_models import invoke_with_retry
    
    # Create a mock that always fails
    mock_model = AsyncMock()
    
    expired_error = ClientError(
        {'Error': {'Code': 'ExpiredTokenException', 'Message': 'Token expired'}},
        'Converse'
    )
    
    mock_model.ainvoke = AsyncMock(side_effect=expired_error)
    
    # Mock get_model to return our mock model
    with patch('app.core.llm.chat_models.get_model', return_value=mock_model):
        try:
            result = await invoke_with_retry(mock_model, [{"role": "user", "content": "test"}], efficient=True, max_retries=1)
            assert False, "Should have raised ClientError"
        except ClientError as e:
            assert e.response['Error']['Code'] == 'ExpiredTokenException'
            assert mock_model.ainvoke.call_count == 2  # Initial + 1 retry
            print("✅ Test 3 passed: invoke_with_retry raises after max retries")


async def test_invoke_with_retry_other_error():
    """Test that invoke_with_retry doesn't retry on other errors"""
    from app.core.llm.chat_models import invoke_with_retry
    
    # Create a mock that fails with a different error
    mock_model = AsyncMock()
    
    other_error = ClientError(
        {'Error': {'Code': 'ValidationException', 'Message': 'Invalid input'}},
        'Converse'
    )
    
    mock_model.ainvoke = AsyncMock(side_effect=other_error)
    
    try:
        result = await invoke_with_retry(mock_model, [{"role": "user", "content": "test"}], efficient=True, max_retries=1)
        assert False, "Should have raised ClientError"
    except ClientError as e:
        assert e.response['Error']['Code'] == 'ValidationException'
        assert mock_model.ainvoke.call_count == 1  # No retry for other errors
        print("✅ Test 4 passed: invoke_with_retry doesn't retry on other errors")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING EXPIREDTOKENEXCEPTION RETRY LOGIC")
    print("="*60 + "\n")
    
    await test_invoke_with_retry_success()
    await test_invoke_with_retry_token_expiry_then_success()
    await test_invoke_with_retry_max_retries_exceeded()
    await test_invoke_with_retry_other_error()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
