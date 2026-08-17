#!/usr/bin/env python3
"""
##############################################################################
KUBECTL SAFETY WARNING - CRITICAL
##############################################################################
This script contains kubectl commands that can MODIFY or DELETE resources

DANGEROUS OPERATIONS DETECTED:
- This script can affect live Kubernetes clusters
- Changes may be irreversible
- Always review commands before execution

SAFETY CHECKLIST:
[ ] Verified correct cluster context
[ ] Reviewed all kubectl commands
[ ] Have backups of affected resources
[ ] Tested in non-production environment first
[ ] Notified team of planned changes

To bypass: Set environment variable KUBECTL_SAFETY_BYPASS=yes
Tagged by: kubectl-safety-audit on 2026-02-24 15:38:15
##############################################################################
"""

import os
import sys

if os.getenv('KUBECTL_SAFETY_BYPASS') != 'yes':
    print("\n" + "="*70)
    print("WARNING: This script contains kubectl commands!")
    print(f"Risk Level: CRITICAL")
    print("="*70)
    try:
        import subprocess
        context = subprocess.check_output(['kubectl', 'config', 'current-context'], 
                                         stderr=subprocess.DEVNULL).decode().strip()
        print(f"Current cluster: {context}")
    except:
        print("Current cluster: UNKNOWN")
    print()
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user.")
        sys.exit(1)
    print()

"""
Standalone test suite for kubectl security validation
This extracts just the validation function and tests it
"""

import sys
import shlex
from pathlib import Path

# Read the tools.py file
tools_path = Path(__file__).parent / "app" / "agent_mode" / "tools.py"
if not tools_path.exists():
    print(f"ERROR: Cannot find {tools_path}")
    sys.exit(1)

tools_content = tools_path.read_text()

# Extract the validation function
val_start = tools_content.find("def _validate_kubectl_command")
val_end = tools_content.find("\n\ndef _log_security_event", val_start)
if val_start == -1 or val_end == -1:
    print("ERROR: Cannot find _validate_kubectl_command function")
    sys.exit(1)

validation_func_code = tools_content[val_start:val_end]

# Execute the function in our namespace
exec(validation_func_code, globals())

def test_security():
    """Run comprehensive security tests."""
    
    tests = [
        # Read-only operations (SHOULD PASS)
        ("kubectl get pods", True),
        ("kubectl get deployments -n default", True),
        ("kubectl describe pod mypod", True),
        ("kubectl logs mypod", True),
        ("kubectl logs mypod --tail=100", True),
        ("kubectl top nodes", True),
        ("kubectl version", True),
        ("kubectl cluster-info", True),
        ("kubectl get pods --all-namespaces", True),
        ("kubectl describe node mynode", True),
        
        # Write operations (SHOULD FAIL)
        ("kubectl delete pod mypod", False),
        ("kubectl delete deployment myapp", False),
        ("kubectl apply -f config.yaml", False),
        ("kubectl create deployment test --image=nginx", False),
        ("kubectl patch pod mypod -p '{}'", False),
        ("kubectl edit deployment myapp", False),
        ("kubectl scale deployment myapp --replicas=0", False),
        ("kubectl rollout restart deployment/myapp", False),
        ("kubectl exec -it mypod -- sh", False),
        ("kubectl run test --image=nginx", False),
        ("kubectl label pod mypod app=test", False),
        ("kubectl annotate pod mypod desc=test", False),
        ("kubectl drain node1", False),
        ("kubectl cordon node1", False),
        ("kubectl taint nodes node1 key=value:NoSchedule", False),
        ("kubectl port-forward pod/mypod 8080:80", False),
        ("kubectl proxy", False),
        ("kubectl expose deployment myapp --port=80", False),
        ("kubectl set image deployment/myapp nginx=nginx:1.19", False),
        ("kubectl replace -f config.yaml", False),
        ("kubectl autoscale deployment myapp --min=2 --max=10", False),
        ("kubectl cp mypod:/tmp/file ./file", False),
        ("kubectl attach mypod", False),
        
        # Bypass attempts (SHOULD ALL FAIL)
        ("kubectl get pods; kubectl delete pods --all", False),
        ("kubectl get pods && kubectl delete pod mypod", False),
        ("kubectl get pods | grep Running | kubectl delete", False),
        ("kubectl get pods --force", False),
        ("kubectl describe pod mypod --cascade=true", False),
        ("kubectl logs mypod -- rm -rf /", False),
        ("kubectl get pods -c bash", False),
        ("kubectl describe pod -- sh -c 'delete something'", False),
        ("kubectl exec mypod -- /bin/sh", False),
        ("kubectl run test --image=nginx --rm -it -- sh", False),
        ("kubectl get pods `delete something`", False),
        ("kubectl delete pod --all", False),
        
        # Edge cases
        ("kubectl", False),  # No verb
        ("kubectl invalid-verb pods", False),  # Invalid verb
    ]
    
    passed = 0
    failed = 0
    failures = []
    
    print("\n" + "="*80)
    print("🔒 KUBECTL SECURITY VALIDATION TEST SUITE")
    print("="*80 + "\n")
    
    for command, should_pass in tests:
        is_valid, error_msg = _validate_kubectl_command(command)
        
        if is_valid == should_pass:
            passed += 1
        else:
            failed += 1
            failure_info = {
                'command': command,
                'expected': 'ALLOW' if should_pass else 'BLOCK',
                'got': 'ALLOW' if is_valid else 'BLOCK',
                'error_msg': error_msg
            }
            failures.append(failure_info)
    
    print(f"\n" + "="*80)
    print(f"📊 TEST RESULTS: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("="*80 + "\n")
    
    if failures:
        print("❌ FAILED TESTS:\n")
        for f in failures:
            print(f"  Command: {f['command']}")
            print(f"  Expected: {f['expected']}, Got: {f['got']}")
            if f['error_msg']:
                print(f"  Reason: {f['error_msg']}")
            print()
    
    if failed == 0:
        print("✅ ALL SECURITY TESTS PASSED!")
        print("🔒 kubectl write protection is functioning correctly.\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED - SECURITY ISSUE!")
        print("⚠️  DO NOT deploy until all tests pass.\n")
        return 1

if __name__ == "__main__":
    sys.exit(test_security())
