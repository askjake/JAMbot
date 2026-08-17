#!/usr/bin/env python3
"""
Comprehensive test to verify all imports work correctly.
This must pass BEFORE deploying.
"""
import sys
import os

# IMPORTANT: Test against the LOCAL repo where we're deploying
# NOT the sandbox repo
LOCAL_REPO = os.path.expanduser('~/dish-chat')
sys.path.insert(0, LOCAL_REPO)

def test_imports():
    """Test that all imports work correctly"""
    print("="*60)
    print("TESTING IMPORTS")
    print("="*60)
    print(f"Testing against: {LOCAL_REPO}")
    
    try:
        # Test 1: Import from core.llm module
        print("\n[1/5] Testing: from app.core.llm import get_model, invoke_with_retry")
        from app.core.llm import get_model, invoke_with_retry
        print("✅ PASS: Both functions imported successfully")
        
        # Test 2: Verify functions exist and are callable
        print("\n[2/5] Testing: Functions are callable")
        assert callable(get_model), "get_model is not callable"
        assert callable(invoke_with_retry), "invoke_with_retry is not callable"
        print("✅ PASS: Both functions are callable")
        
        # Test 3: Check function signatures
        print("\n[3/5] Testing: Function signatures")
        import inspect
        
        get_model_sig = inspect.signature(get_model)
        print(f"  get_model signature: {get_model_sig}")
        assert 'efficient' in get_model_sig.parameters
        assert 'force_refresh' in get_model_sig.parameters
        print("✅ PASS: get_model has correct parameters")
        
        invoke_sig = inspect.signature(invoke_with_retry)
        print(f"  invoke_with_retry signature: {invoke_sig}")
        assert 'model' in invoke_sig.parameters
        assert 'messages' in invoke_sig.parameters
        assert 'efficient' in invoke_sig.parameters
        assert 'max_retries' in invoke_sig.parameters
        print("✅ PASS: invoke_with_retry has correct parameters")
        
        # Test 4: Test that journal service can import
        print("\n[4/5] Testing: Journal service imports")
        # This simulates what happens when the app starts
        from app.journal.service import JournalService
        print("✅ PASS: JournalService imports successfully")
        
        # Test 5: Verify the service has the method
        print("\n[5/5] Testing: JournalService has generate_journal_entry method")
        assert hasattr(JournalService, 'generate_journal_entry')
        print("✅ PASS: JournalService has generate_journal_entry method")
        
        print("\n" + "="*60)
        print("ALL IMPORT TESTS PASSED ✅")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
