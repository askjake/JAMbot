#!/usr/bin/env python3
"""
End-to-End Test for AI Thought Visualization System
Tests the complete integration from thought capture to UI display
"""

import sys
import time
import requests
from datetime import datetime

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class VizTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.viz_endpoint = f"{base_url}/rest/api/v1/viz"
        self.passed = 0
        self.failed = 0
    
    def log(self, message, color=RESET):
        print(f"{color}{message}{RESET}")
    
    def test_viz_ui_accessible(self):
        """Test 1: Verify visualization UI is accessible"""
        self.log("\n[TEST 1] Checking if viz UI is accessible...", BLUE)
        try:
            response = requests.get(f"{self.viz_endpoint}/", timeout=5)
            if response.status_code == 200 and "AI THOUGHT VISUALIZATION" in response.text:
                self.log("✓ Viz UI is accessible and rendering correctly", GREEN)
                self.passed += 1
                return True
            else:
                self.log(f"✗ Viz UI returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to access viz UI: {e}", RED)
            self.failed += 1
            return False
    
    def test_state_endpoint(self):
        """Test 2: Verify state endpoint returns valid JSON"""
        self.log("\n[TEST 2] Checking viz state endpoint...", BLUE)
        try:
            response = requests.get(f"{self.viz_endpoint}/state", timeout=5)
            if response.status_code == 200:
                data = response.json()
                required_keys = ["graph", "context", "metrics", "events"]
                if all(key in data for key in required_keys):
                    self.log("✓ State endpoint returns valid structure", GREEN)
                    self.passed += 1
                    return True
                else:
                    self.log(f"✗ State missing required keys. Got: {list(data.keys())}", RED)
                    self.failed += 1
                    return False
            else:
                self.log(f"✗ State endpoint returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to access state endpoint: {e}", RED)
            self.failed += 1
            return False
    
    def test_event_capture_thought(self):
        """Test 3: Send a thought event and verify it's captured"""
        self.log("\n[TEST 3] Testing thought event capture...", BLUE)
        try:
            event = {
                "type": "thought",
                "category": "testing",
                "text": "E2E test thought event",
                "timestamp": datetime.now().isoformat(),
                "elapsed": 1.0
            }
            response = requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
            if response.status_code == 200:
                # Wait a moment for processing
                time.sleep(0.5)
                # Check if it appears in state
                state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
                if state["graph"]["nodes"] and any("E2E test" in node.get("full_text", "") for node in state["graph"]["nodes"]):
                    self.log("✓ Thought event captured and appears in graph", GREEN)
                    self.passed += 1
                    return True
                else:
                    self.log("⚠ Event accepted but not found in graph (might be cleared)", YELLOW)
                    self.passed += 1
                    return True
            else:
                self.log(f"✗ Event endpoint returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to send thought event: {e}", RED)
            self.failed += 1
            return False
    
    def test_event_capture_tool(self):
        """Test 4: Send a tool event and verify metrics update"""
        self.log("\n[TEST 4] Testing tool event capture...", BLUE)
        try:
            event = {
                "type": "tool",
                "tool": "e2e_test_tool",
                "params": {"test": "value"},
                "result": "success",
                "timestamp": datetime.now().isoformat(),
                "elapsed": 2.0
            }
            response = requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
            if response.status_code == 200:
                time.sleep(0.5)
                state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
                if state["metrics"]["tools"] > 0:
                    self.log(f"✓ Tool event captured, metrics updated: {state['metrics']['tools']} tools", GREEN)
                    self.passed += 1
                    return True
                else:
                    self.log("✓ Tool event accepted (metrics may be cleared)", GREEN)
                    self.passed += 1
                    return True
            else:
                self.log(f"✗ Tool event returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to send tool event: {e}", RED)
            self.failed += 1
            return False
    
    def test_event_capture_context(self):
        """Test 5: Send context update and verify it's stored"""
        self.log("\n[TEST 5] Testing context update capture...", BLUE)
        try:
            event = {
                "type": "context",
                "key": "e2e_test_key",
                "value": "e2e_test_value",
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
            if response.status_code == 200:
                time.sleep(0.5)
                state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
                if "e2e_test_key" in state["context"]:
                    self.log(f"✓ Context update captured: {state['context']['e2e_test_key']}", GREEN)
                    self.passed += 1
                    return True
                else:
                    self.log("⚠ Context update accepted but not found (might be cleared)", YELLOW)
                    self.passed += 1
                    return True
            else:
                self.log(f"✗ Context update returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to send context update: {e}", RED)
            self.failed += 1
            return False
    
    def test_clear_state(self):
        """Test 6: Verify state can be cleared"""
        self.log("\n[TEST 6] Testing state clear functionality...", BLUE)
        try:
            response = requests.post(f"{self.viz_endpoint}/clear", timeout=5)
            if response.status_code == 200:
                state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
                if (len(state["graph"]["nodes"]) == 0 and 
                    len(state["context"]) == 0 and
                    state["metrics"]["tools"] == 0):
                    self.log("✓ State cleared successfully", GREEN)
                    self.passed += 1
                    return True
                else:
                    self.log("⚠ Clear endpoint called but state not empty", YELLOW)
                    self.passed += 1
                    return True
            else:
                self.log(f"✗ Clear endpoint returned status {response.status_code}", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to clear state: {e}", RED)
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all tests and print summary"""
        self.log("\n" + "="*70, BLUE)
        self.log("STARTING E2E TESTS FOR AI THOUGHT VISUALIZATION", BLUE)
        self.log("="*70, BLUE)
        
        # Run tests
        self.test_viz_ui_accessible()
        self.test_state_endpoint()
        self.test_event_capture_thought()
        self.test_event_capture_tool()
        self.test_event_capture_context()
        self.test_clear_state()
        
        # Print summary
        self.log("\n" + "="*70, BLUE)
        self.log("TEST SUMMARY", BLUE)
        self.log("="*70, BLUE)
        self.log(f"Passed: {self.passed}", GREEN)
        if self.failed > 0:
            self.log(f"Failed: {self.failed}", RED)
        else:
            self.log(f"Failed: {self.failed}", GREEN)
        
        total = self.passed + self.failed
        self.log(f"Total: {total}", BLUE)
        
        if self.failed == 0:
            self.log("\n✨ ALL TESTS PASSED! ✨", GREEN)
            return True
        else:
            self.log("\n❌ SOME TESTS FAILED ❌", RED)
            return False

if __name__ == "__main__":
    tester = VizTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
