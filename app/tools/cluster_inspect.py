# app/tools/cluster_inspect.py
import logging
import subprocess
import os
from typing import Dict, List

from langchain.tools import tool

from app.agent_mode.thought_interceptor import interceptor

logger = logging.getLogger(__name__)

# Curated, read-only commands – expanded for more functionality
ALLOWED_TASKS: Dict[str, List[str]] = {
    # Sentry commands
    "sentry_claims": ["kubectl", "get", "sentryclaims", "-A", "-o", "wide"],
    "sentry_pods": ["kubectl", "get", "pods", "-n", "sentry", "-o", "wide"],
    "sentry_all_pods": ["kubectl", "get", "pods", "-A", "-o", "wide"],
    "sentry_helm": ["helm", "list", "-A"],
    
    # General kubectl commands
    "get_namespaces": ["kubectl", "get", "namespaces"],
    "get_nodes": ["kubectl", "get", "nodes", "-o", "wide"],
    "get_deployments": ["kubectl", "get", "deployments", "-A", "-o", "wide"],
    "get_services": ["kubectl", "get", "services", "-A", "-o", "wide"],
    "get_ingress": ["kubectl", "get", "ingress", "-A", "-o", "wide"],
    "get_configmaps": ["kubectl", "get", "configmaps", "-A"],
    "get_secrets": ["kubectl", "get", "secrets", "-A"],
    "get_pvcs": ["kubectl", "get", "pvc", "-A", "-o", "wide"],
    
    # ArgoCD commands
    "argocd_apps": ["kubectl", "get", "applications", "-n", "argocd", "-o", "wide"],
    
    # ArgoCD CLI commands (read-only)
    "argocd_app_list": ["argocd", "app", "list"],
    "argocd_app_get": ["argocd", "app", "get"],
    "argocd_app_diff": ["argocd", "app", "diff"],
    "argocd_app_history": ["argocd", "app", "history"],
    "argocd_app_manifests": ["argocd", "app", "manifests"],
    "argocd_app_resources": ["argocd", "app", "resources"],
    "argocd_app_logs": ["argocd", "app", "logs"],
    "argocd_cluster_list": ["argocd", "cluster", "list"],
    "argocd_repo_list": ["argocd", "repo", "list"],
    "argocd_proj_list": ["argocd", "proj", "list"],
    "argocd_version": ["argocd", "version", "--client"],
    
    # Helm commands
    "helm_list_all": ["helm", "list", "-A"],
    
    # Context commands
    "get_contexts": ["kubectl", "config", "get-contexts"],
    "current_context": ["kubectl", "config", "current-context"],
}


def _run(cmd: List[str]) -> str:
    """Execute a command and return output."""
    try:
        # Add ~/bin to PATH for argocd CLI
        env = os.environ.copy()
        home = os.path.expanduser("~")
        env["PATH"] = f"{home}/bin:{env.get('PATH', '')}"
        
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except Exception as e:
        logger.error("cluster_inspect failed to run %r: %s", cmd, e, exc_info=True)
        return f"cluster_inspect: error running {cmd!r}: {e}"

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    
    if proc.returncode != 0:
        return (
            f"cluster_inspect: command {cmd!r} exited {proc.returncode}\n"
            f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
        )

    if err:
        return f"STDOUT:\n{out}\n\nSTDERR (non-fatal):\n{err}"

    return out or "(no output)"


def _run_kubectl_namespace(namespace: str, resource: str, extra_args: List[str] = None) -> str:
    """Run kubectl get for a specific namespace."""
    cmd = ["kubectl", "get", resource, "-n", namespace]
    if extra_args:
        cmd.extend(extra_args)
    else:
        cmd.extend(["-o", "wide"])
    return _run(cmd)


def _run_kubectl_describe(namespace: str, resource_type: str, resource_name: str) -> str:
    """Run kubectl describe for a specific resource."""
    cmd = ["kubectl", "describe", resource_type, resource_name, "-n", namespace]
    return _run(cmd)


def _run_kubectl_logs(namespace: str, pod_name: str, tail: int = 100, container: str = None) -> str:
    """Get logs from a pod."""
    cmd = ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={tail}"]
    if container:
        cmd.extend(["-c", container])
    return _run(cmd)


@tool("cluster_inspect")
def cluster_inspect(task: str) -> str:
    """
    Run *read-only* cluster inspections via curated commands.

    Supported tasks:
    - "list sentry claims" - List Sentry claims
    - "list sentry pods" - List pods in sentry namespace
    - "list all pods" - List all pods in all namespaces
    - "list sentry helm releases" - List Helm releases
    - "list namespaces" - List all namespaces
    - "list nodes" - List cluster nodes
    - "list deployments" - List all deployments
    - "list services" - List all services
    - "list ingress" - List all ingress resources
    - "argocd app list" - List all ArgoCD applications (via CLI)
    - "argocd app get <app-name>" - Get application details
    - "argocd app diff <app-name>" - Show application diff
    - "argocd app history <app-name>" - Show deployment history
    - "argocd cluster list" - List registered clusters
    - "argocd repo list" - List configured repositories
    - "argocd proj list" - List ArgoCD projects
    - "list argocd apps" - List ArgoCD applications
    - "get contexts" - List kubectl contexts
    - "current context" - Show current kubectl context
    - "pods in <namespace>" - List pods in specific namespace
    - "describe pod <pod-name> in <namespace>" - Describe a pod
    - "logs from <pod-name> in <namespace>" - Get pod logs (last 100 lines)
    - "logs from <pod-name> in <namespace> tail 500" - Get pod logs with custom tail

    Returns raw text output for the agent to interpret.
    """
    interceptor.tool_call("cluster_inspect", params={"task": task})
    interceptor.thought(f"Inspecting cluster: {task}", "tool")
    
    t = task.lower().strip()
    
    # Parse and route the task
    result = None
    
    # Predefined tasks
    if "sentry" in t and "claim" in t:
        result = _run(ALLOWED_TASKS["sentry_claims"])
    elif "sentry" in t and ("pod" in t or "deployment" in t):
        result = _run(ALLOWED_TASKS["sentry_pods"])
    elif "all" in t and "pod" in t:
        result = _run(ALLOWED_TASKS["sentry_all_pods"])
    elif "sentry" in t and "helm" in t:
        result = _run(ALLOWED_TASKS["sentry_helm"])
    elif "list namespaces" in t or "get namespaces" in t:
        result = _run(ALLOWED_TASKS["get_namespaces"])
    elif "list nodes" in t or "get nodes" in t:
        result = _run(ALLOWED_TASKS["get_nodes"])
    elif "list deployments" in t:
        result = _run(ALLOWED_TASKS["get_deployments"])
    elif "list services" in t:
        result = _run(ALLOWED_TASKS["get_services"])
    elif "list ingress" in t:
        result = _run(ALLOWED_TASKS["get_ingress"])
    elif "argocd" in t and "app" in t:
        result = _run(ALLOWED_TASKS["argocd_apps"])
    elif "helm" in t and "list" in t:
        result = _run(ALLOWED_TASKS["helm_list_all"])
    elif "get contexts" in t or "list contexts" in t:
        result = _run(ALLOWED_TASKS["get_contexts"])
    elif "current context" in t:
        result = _run(ALLOWED_TASKS["current_context"])
    
    # ArgoCD CLI commands
    elif 'argocd' in t:
        if 'list apps' in t or 'app list' in t:
            result = _run(ALLOWED_TASKS['argocd_app_list'])
        elif 'app get' in t:
            # Extract app name: "argocd app get myapp"
            parts = t.split('app get')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_get'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'app diff' in t:
            parts = t.split('app diff')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_diff'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'app history' in t:
            parts = t.split('app history')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_history'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'app manifests' in t:
            parts = t.split('app manifests')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_manifests'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'app resources' in t:
            parts = t.split('app resources')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_resources'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'app logs' in t:
            parts = t.split('app logs')
            if len(parts) == 2 and parts[1].strip():
                app_name = parts[1].strip()
                cmd = ALLOWED_TASKS['argocd_app_logs'].copy()
                cmd.append(app_name)
                result = _run(cmd)
        elif 'cluster list' in t or 'list clusters' in t:
            result = _run(ALLOWED_TASKS['argocd_cluster_list'])
        elif 'repo list' in t or 'list repos' in t:
            result = _run(ALLOWED_TASKS['argocd_repo_list'])
        elif 'proj list' in t or 'project list' in t or 'list projects' in t:
            result = _run(ALLOWED_TASKS['argocd_proj_list'])
        elif 'version' in t:
            result = _run(ALLOWED_TASKS['argocd_version'])
    
    # Dynamic namespace queries
    elif "pods in" in t:
        # Extract namespace: "pods in chatbot-agent"
        parts = t.split("pods in")
        if len(parts) == 2:
            namespace = parts[1].strip()
            result = _run_kubectl_namespace(namespace, "pods")
    
    elif "describe pod" in t and " in " in t:
        # Extract pod and namespace: "describe pod my-pod in my-namespace"
        parts = t.split("describe pod")[1].split(" in ")
        if len(parts) == 2:
            pod_name = parts[0].strip()
            namespace = parts[1].strip()
            result = _run_kubectl_describe(namespace, "pod", pod_name)
    
    elif "logs from" in t and " in " in t:
        # Extract pod and namespace: "logs from my-pod in my-namespace"
        # Optional: "logs from my-pod in my-namespace tail 500"
        parts = t.split("logs from")[1].split(" in ")
        if len(parts) == 2:
            pod_part = parts[0].strip()
            namespace_part = parts[1].strip()
            
            # Check for tail parameter
            tail = 100
            if "tail" in namespace_part:
                ns_parts = namespace_part.split("tail")
                namespace = ns_parts[0].strip()
                try:
                    tail = int(ns_parts[1].strip())
                except:
                    tail = 100
            else:
                namespace = namespace_part
            
            result = _run_kubectl_logs(namespace, pod_part, tail=tail)
    
    # If no match found
    if result is None:
        result = (
            "cluster_inspect: unsupported task. Try one of:\n"
            "- 'list sentry claims/pods/helm releases'\n"
            "- 'list namespaces/nodes/deployments/services/ingress'\n"
            "- 'list argocd apps'\n"
            "- argocd app list/get/diff/history - ArgoCD CLI commands\n"
            "- 'get contexts' or 'current context'\n"
            "- 'pods in <namespace>'\n"
            "- 'describe pod <pod-name> in <namespace>'\n"
            "- 'logs from <pod-name> in <namespace> [tail N]'"
        )
        interceptor.tool_call("cluster_inspect", result="Unsupported task")
        return result
    
    # Log completion
    if "error" not in result.lower() and "failed" not in result.lower():
        interceptor.tool_call("cluster_inspect", result="Command completed successfully")
    else:
        interceptor.tool_call("cluster_inspect", result="Command failed or returned error")
    
    return result
