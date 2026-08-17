# Backend Production Deployment Ticketing Process

## Overview
This document defines the mandatory ticketing process for all production deployments and architecture changes affecting production services. This process ensures accountability, maintains audit trails, and provides documentation for all changes to production systems.

**Effective Date:** 2026-03-24  
**Authority:** Presented by Dallis Yoder, requested by Jason  
**Source:** Cloned from Enix project standards

---

## Scope

### When This Process Applies
This ticketing process is **MANDATORY** for:

1. **Production Deployments**
   - New service deployments to production environments
   - Version updates or upgrades to existing production services
   - Configuration changes to production systems

2. **Architecture Changes Affecting Production**
   - Infrastructure modifications (Kubernetes, AWS resources, networking)
   - Service mesh or communication pattern changes
   - Database schema or data pipeline modifications
   - Security or access control updates

3. **Production Services In Scope**
   - Set Top Box Health Live
   - Dish Chat
   - All services deployed in production namespaces/clusters
   - Any system with customer-facing impact

4. **Emergency Changes and Hot Fixes**
   - **ALL hot fixes must be documented with a ticket** (even if applied urgently)
   - Emergency infrastructure changes (e.g., Argo Workflow issues)
   - Any change to production, regardless of urgency
   - **Rationale:** Provides audit trail for accountability and historical reference

---

## Required Ticket Information

Every production deployment ticket MUST include the following sections:

### 1. Affected Services
List all services, systems, or components that will be impacted by this change.

**Example:**
```
- api-service (root-cause-analysis namespace)
- log-analyzer (root-cause-analysis namespace)
- AWS S3 bucket: hot-pursuit-log-analysis
- AWS Bedrock inference profile: p30i9173cxce
```

### 2. Reason for Change
Clear explanation of why this change is necessary.

**Example:**
```
Deploying updated log-analyzer with improved Bedrock prompt engineering 
to reduce false positive alerts by 30% as identified in incident INC-12345.
```

### 3. Pre-Plan
Steps to prepare for the deployment, including prerequisite checks and setup.

**Example:**
```
1. Verify current production version: kubectl get deployment log-analyzer -n root-cause-analysis
2. Create backup of current configuration: kubectl get deployment log-analyzer -n root-cause-analysis -o yaml > backup.yaml
3. Confirm ECR image availability: aws ecr describe-images --repository-name root-cause-analysis/log-analyzer --image-ids imageTag=1.2.0
4. Notify #data-solutions channel of maintenance window
5. Verify downstream services (analysis-report-formatter) are healthy
```

### 4. Plan
Detailed step-by-step deployment instructions.

**Example:**
```
1. Update deployment manifest with new image tag (1.2.0)
2. Apply deployment: kubectl apply -f log-analyzer-deployment.yaml
3. Monitor rollout: kubectl rollout status deployment/log-analyzer -n root-cause-analysis
4. Verify health endpoint: curl http://log-analyzer:8003/health
5. Run smoke tests: python tests/smoke_test_log_analyzer.py
6. Monitor logs for 15 minutes: kubectl logs -f deployment/log-analyzer -n root-cause-analysis
7. Verify end-to-end flow with test data
```

### 5. Contingency/Backout Plan
Detailed steps to revert the change if issues occur.

**Example:**
```
IF deployment fails or critical issues occur:

1. Immediately rollback to previous version:
   kubectl rollout undo deployment/log-analyzer -n root-cause-analysis
   
2. Verify rollback success:
   kubectl get deployment log-analyzer -n root-cause-analysis
   kubectl rollout status deployment/log-analyzer -n root-cause-analysis
   
3. Restore from backup if needed:
   kubectl apply -f backup.yaml
   
4. Verify service recovery:
   curl http://log-analyzer:8003/health
   python tests/smoke_test_log_analyzer.py
   
5. Notify #data-solutions channel of rollback
6. Create incident ticket for investigation
7. Estimated rollback time: 5 minutes
```

---

## Approval Workflow

### Two-Stage Approval Required

#### **First Approval (Technical Lead)**
- **Who:** Must be manually assigned to either:
  - **Jim** (Infrastructure/Platform decisions)
  - **Jared** (Application/Service decisions)
- **Review Focus:**
  - Technical feasibility and completeness
  - Risk assessment
  - Plan and backout plan adequacy
  - Resource availability

#### **Second Approval (Management)**
- **Who:** Automatically assigned to **Jason** (mandatory review)
- **Review Focus:**
  - Business impact assessment
  - Change timing and coordination
  - Cross-team communication
  - Final authorization

### Approval Process Flow
```
Ticket Created
    ↓
First Approval: Jim OR Jared (manual assignment)
    ↓
Second Approval: Jason (automatic assignment)
    ↓
Approved → Proceed with Deployment
```

---

## Emergency Changes and Hot Fixes

### Special Handling for Urgent Changes

Even for emergency situations, **documentation is mandatory**:

1. **Priority Order:**
   - System stability and recovery comes first
   - Documentation comes immediately after (not before)

2. **Emergency Ticket Requirements:**
   - Create ticket as soon as stability is restored
   - Document what was changed and why
   - Include timestamp of emergency action
   - Note who performed the change
   - Provide post-mortem summary

3. **Rationale:**
   - Maintains audit trail for compliance
   - Provides accountability for changes
   - Creates historical reference for future incidents
   - Enables pattern analysis and prevention

**Example Emergency Ticket:**
```
Summary: Emergency Rollback of api-service Due to Memory Leak

Affected Services: api-service (production)

Reason: Production api-service experiencing OOMKilled errors, impacting customer access

Emergency Actions Taken (2026-03-24 14:32 UTC):
- Rolled back api-service from v2.1.0 to v2.0.5 (James Fitzgerald)
- Restarted all pods to clear memory state
- Verified service recovery

Pre-Plan: N/A (Emergency response)

Plan Executed:
1. Identified OOMKilled errors via kubectl describe pod
2. Rolled back: kubectl rollout undo deployment/api-service -n production
3. Monitored recovery: kubectl get pods -n production -w
4. Confirmed health endpoints responding

Contingency: Rolled back successfully to last known good version

Post-Mortem: Memory leak introduced in v2.1.0, requires investigation before redeployment
```

---

## Ticket Template

Use this template when creating production deployment tickets:

```markdown
## Affected Services
- [List all affected services, components, AWS resources]

## Reason for Change
[Clear explanation of why this change is necessary]

## Pre-Plan
1. [Preparation step 1]
2. [Preparation step 2]
...

## Plan
1. [Deployment step 1]
2. [Deployment step 2]
...

## Contingency/Backout Plan
1. [Rollback step 1]
2. [Rollback step 2]
...
Estimated rollback time: [X minutes]

## Approvals Required
- [ ] First Approval: @Jim OR @Jared (assign manually)
- [ ] Second Approval: @Jason (automatic)

## Additional Notes
[Any relevant context, dependencies, or considerations]
```

---

## Key Principles

1. **Documentation Over Speed (With Exception for Emergencies)**
   - All changes must be documented
   - Even hot fixes require tickets (after stabilization)
   - No undocumented changes to production

2. **Accountability and Audit Trail**
   - Every change has a clear owner
   - Approval chain is documented
   - Historical record for compliance and learning

3. **Risk Mitigation**
   - Mandatory backout plans for all changes
   - Pre-deployment verification steps
   - Post-deployment monitoring requirements

4. **Cross-Team Coordination**
   - Technical and management oversight
   - Communication requirements specified
   - Dependencies explicitly documented

---

## Integration with Existing Workflows

- **Argo Workflow Changes:** Follow this ticketing process
- **Kubernetes Manifests:** Include ticket reference in commit messages
- **Infrastructure as Code (Terraform):** Reference ticket in PR description
- **Emergency Response:** Create ticket immediately after incident resolution

---

## Enforcement

- **Mandatory:** This process is required for all production changes
- **No Exceptions:** Even urgent hot fixes require documentation
- **Audit Reviews:** Jason and leadership will review compliance
- **Non-Compliance:** Undocumented production changes will be escalated

---

**Process Owner:** Dallis Yoder  
**Technical Approvers:** Jim, Jared  
**Management Approver:** Jason  
**Last Updated:** 2026-03-24
