"""Tests for query sanitization"""
from app.tools.query_sanitizer import contains_sensitive_info, should_block_query

def test_sensitive_queries():
    # Should detect sensitive info
    assert contains_sensitive_info("john.doe@dish.com password reset")[0] == True
    assert contains_sensitive_info("RAIL-cluster-prod configuration")[0] == True
    assert contains_sensitive_info("192.168.1.100 server status")[0] == True
    assert contains_sensitive_info("internal-api.dish.com endpoint")[0] == True
    
    # Should be safe
    assert contains_sensitive_info("weather in Denver")[0] == False
    assert contains_sensitive_info("kubernetes best practices")[0] == False
    assert contains_sensitive_info("python async programming")[0] == False

def test_blocking():
    # Should block
    assert should_block_query("user@dish.com credentials")[0] == True
    assert should_block_query("api_key=abc123")[0] == True
    
    # Should not block (warning only)
    assert should_block_query("DISH RAIL cluster")[0] == False
    assert should_block_query("weather forecast")[0] == False

if __name__ == "__main__":
    test_sensitive_queries()
    test_blocking()
    print("✅ All sanitization tests passed!")
