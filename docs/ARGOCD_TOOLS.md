# ArgoCD CLI Tools Integration

## Overview
The agent now has comprehensive ArgoCD CLI support integrated into the cluster_inspect tool.
This enables read-only inspection and monitoring of ArgoCD applications, clusters, repositories, and projects.

## Installation
- Location: /home/montjac/bin/argocd
- Version: v3.3.5+b8b5ea6
- Added to PATH: Automatically included in cluster_inspect tool execution

## Available Commands via cluster_inspect

### List Operations
- argocd app list - List all applications
- argocd cluster list - List registered clusters
- argocd repo list - List configured repositories
- argocd proj list - List ArgoCD projects
- argocd version - Show ArgoCD version

### Application Details (requires app name)
- argocd app get <name> - Get application details
- argocd app diff <name> - Show application diff
- argocd app history <name> - Show deployment history
- argocd app manifests <name> - Show application manifests
- argocd app resources <name> - List application resources
- argocd app logs <name> - Show application logs

## When to Use ArgoCD vs K8s

Use ArgoCD CLI when:
- Checking application sync status
- Viewing deployment history
- Comparing live state with desired state
- Inspecting ArgoCD-specific resources

Use K8s tools when:
- Inspecting individual pods/services
- Getting detailed resource specs
- Viewing raw YAML manifests

## Security Model

ALLOWED (read-only):
- app list/get/diff/history/manifests/resources/logs
- cluster/repo/proj list
- version

BLOCKED (write operations):
- app sync/delete/rollback/set/create/patch
- repo/cluster add/rm

## Agent Usage Example

cluster_inspect("argocd app list")
cluster_inspect("argocd app get myapp")
cluster_inspect("argocd app diff myapp")

## Changelog
2026-03-26: Initial ArgoCD CLI integration
