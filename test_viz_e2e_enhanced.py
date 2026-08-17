#!/usr/bin/env python3
"""
Enhanced E2E Test for AI Thought Visualization System
Tests metrics accumulation, reasoning graph, and all data alignment
"""

import sys
import time
import requests
from datetime import datetime

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class EnhancedVizTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.viz_endpoint = f"{base_url}/rest/api/v1/viz"
        self.passed = 0
        self.failed = 0
    
    def log(self, message, color=RESET):
        print(f"{color}{message}{RESET}")
    
    def test_metrics_accumulation(self):
        """Test that metrics accumulate instead of replacing"""
        self.log("\n[TEST] Checking metrics accumulation...", BLUE)
        
        try:
            # Clear state first
            requests.post(f"{self.viz_endpoint}/clear", timeout=5)
            time.sleep(0.5)
            
            # Send multiple token metrics
            for i in range(3):
                event = {
                    "type": "metric",
                    "metric": "tokens",
                    "value": 100,
                    "timestamp": datetime.now().isoformat()
                }
                requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
                time.sleep(0.2)
            
            # Check if tokens accumulated (should be 300, not 100)
            state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
            token_count = state["metrics"].get("tokens", 0)
            
            if token_count == 300:
                self.log(f"✓ Metrics accumulate correctly: {token_count} tokens", GREEN)
                self.passed += 1
                return True
            elif token_count == 100:
                self.log(f"✗ Metrics are REPLACING instead of accumulating: {token_count} tokens", RED)
                self.failed += 1
                return False
            else:
                self.log(f"⚠ Unexpected token count: {token_count}", YELLOW)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to test metric accumulation: {e}", RED)
            self.failed += 1
            return False
    
    def test_time_metrics(self):
        """Test that time metrics are tracked"""
        self.log("\n[TEST] Checking time metrics...", BLUE)
        
        try:
            # Send time metric
            event = {
                "type": "metric",
                "metric": "time",
                "value": 2.5,
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
            time.sleep(0.5)
            
            state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
            time_value = state["metrics"].get("time", 0)
            
            if time_value > 0:
                self.log(f"✓ Time metrics are tracked: {time_value}s", GREEN)
                self.passed += 1
                return True
            else:
                self.log(f"✗ Time metrics are NOT being tracked", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to test time metrics: {e}", RED)
            self.failed += 1
            return False
    
    def test_reasoning_graph_nodes(self):
        """Test that reasoning graph nodes are created with proper IDs"""
        self.log("\n[TEST] Checking reasoning graph nodes...", BLUE)
        
        try:
            # Clear state
            requests.post(f"{self.viz_endpoint}/clear", timeout=5)
            time.sleep(0.5)
            
            # Send multiple thoughts
            for i in range(5):
                event = {
                    "type": "thought",
                    "category": "thinking",
                    "text": f"Test thought {i+1}",
                    "timestamp": datetime.now().isoformat(),
                    "elapsed": i * 0.5
                }
                requests.post(f"{self.viz_endpoint}/event", json=event, timeout=5)
                time.sleep(0.1)
            
            state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
            nodes = state["graph"]["nodes"]
            links = state["graph"]["links"]
            
            if len(nodes) == 5:
                self.log(f"✓ All 5 thought nodes created", GREEN)
            else:
                self.log(f"⚠ Expected 5 nodes, got {len(nodes)}", YELLOW)
            
            if len(links) == 4:  # 5 nodes = 4 links
                self.log(f"✓ All 4 links created between nodes", GREEN)
            else:
                self.log(f"⚠ Expected 4 links, got {len(links)}", YELLOW)
            
            # Check if nodes have proper IDs
            node_ids = [n.get("id") for n in nodes]
            if len(set(node_ids)) == len(nodes):
                self.log(f"✓ All nodes have unique IDs", GREEN)
                self.passed += 1
                return True
            else:
                self.log(f"✗ Nodes have duplicate IDs", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to test reasoning graph: {e}", RED)
            self.failed += 1
            return False
    
    def test_all_data_points_aligned(self):
        """Test that all data points work together"""
        self.log("\n[TEST] Checking all data points alignment...", BLUE)
        
        try:
            # Clear state
            requests.post(f"{self.viz_endpoint}/clear", timeout=5)
            time.sleep(0.5)
            
            # Simulate a complete agent interaction
            # 1. Thought
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "thought",
                "category": "thinking",
                "text": "Analyzing user request",
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            # 2. Tool call
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "tool",
                "tool": "web_search",
                "params": {"query": "test"},
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            # 3. Decision
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "decision",
                "text": "Use search results",
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            # 4. Metrics
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "metric",
                "metric": "tokens",
                "value": 150,
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "metric",
                "metric": "time",
                "value": 1.5,
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            # 5. Context
            requests.post(f"{self.viz_endpoint}/event", json={
                "type": "context",
                "key": "status",
                "value": "complete",
                "timestamp": datetime.now().isoformat()
            }, timeout=5)
            
            time.sleep(0.5)
            
            # Check all data points
            state = requests.get(f"{self.viz_endpoint}/state", timeout=5).json()
            
            checks = {
                "reasoning_graph": len(state["graph"]["nodes"]) >= 2,  # thought + decision
                "tools_tracked": state["metrics"]["tools"] >= 1,
                "tokens_tracked": state["metrics"]["tokens"] == 150,
                "time_tracked": state["metrics"]["time"] == 1.5,
                "context_stored": "status" in state["context"]
            }
            
            all_passed = all(checks.values())
            
            self.log("\nData point checks:", BLUE)
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                color = GREEN if passed else RED
                self.log(f"  {status} {check}: {passed}", color)
            
            if all_passed:
                self.log("\n✓ All data points properly aligned!", GREEN)
                self.passed += 1
                return True
            else:
                self.log("\n✗ Some data points not aligned", RED)
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"✗ Failed to test data alignment: {e}", RED)
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all enhanced tests"""
        self.log("\n" + "="*70, BLUE)
        self.log("ENHANCED E2E TESTS FOR VIZ DATA ALIGNMENT", BLUE)
        self.log("="*70, BLUE)
        
        self.test_metrics_accumulation()
        self.test_time_metrics()
        self.test_reasoning_graph_nodes()
        self.test_all_data_points_aligned()
        
        # Summary
        self.log("\n" + "="*70, BLUE)
        self.log("TEST SUMMARY", BLUE)
        self.log("="*70, BLUE)
        self.log(f"Passed: {self.passed}", GREEN)
        if self.failed > 0:
            self.log(f"Failed: {self.failed}", RED)
        else:
            self.log(f"Failed: {self.failed}", GREEN)
        
        if self.failed == 0:
            self.log("\n✨ ALL TESTS PASSED! ✨", GREEN)
            return True
        else:
            self.log("\n❌ SOME TESTS FAILED ❌", RED)
            return False

if __name__ == "__main__":
    tester = EnhancedVizTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
