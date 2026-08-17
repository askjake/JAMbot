# Sentry Cluster Upgrade Plan - Comprehensive Runbook

**Cluster:** sentry-xx-eks-skd5s-d94nn  
**Region:** us-west-2  
**Created:** 2026-01-22  
**Estimated Total Duration:** 4-6 hours (spread across maintenance windows)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [General Prerequisites](#general-prerequisites)
3. [Upgrade Order](#upgrade-order)
4. [Component Runbooks](#component-runbooks)
   - [1. EKS Node OS Upgrade](#1-eks-node-os-upgrade)
   - [2. Sentry Application Pods](#2-sentry-application-pods)
   - [3. Memcached](#3-memcached)
   - [4. RabbitMQ](#4-rabbitmq)
   - [5. ClickHouse](#5-clickhouse)
   - [6. Zookeeper](#6-zookeeper)
5. [Post-Upgrade Validation](#post-upgrade-validation)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

This runbook provides step-by-step instructions for upgrading all components of your Sentry cluster. The upgrades are ordered by priority and dependency, with the lowest-risk upgrades first.

### Risk Assessment

| Component | Complexity | Downtime Risk | Can Rollback? |
|-----------|-----------|---------------|---------------|
| EKS Node OS | Low | Minimal | ✅ Yes |
| Sentry Apps | Low-Medium | None | ✅ Yes |
| Memcached | Low | Low | ✅ Yes |
| RabbitMQ | Medium | Medium | ⚠️ Difficult |
| ClickHouse | High | Medium-High | ❌ Very Difficult |
| Zookeeper | Very High | High | ❌ Very Difficult |

---

## General Prerequisites

Before starting ANY upgrade:

- [ ] Test all upgrades in **ds-test cluster** first
- [ ] Schedule maintenance window during low-traffic period
- [ ] Notify team and stakeholders
- [ ] Ensure monitoring and alerting are active
- [ ] Have AWS console and kubectl access ready
- [ ] Prepare backup storage location
- [ ] Document current versions of all components
- [ ] Have rollback plan ready
- [ ] Ensure you have at least 2 team members available

### Required Tools

```bash
# Verify you have these tools installed
kubectl version --client
helm version
aws --version

# Verify cluster access
kubectl get nodes
kubectl get pods -n sentry
```

---

## Upgrade Order

**IMPORTANT:** Follow this order strictly. Do not skip ahead.

1. ✅ **EKS Node OS** - Foundation layer, lowest risk
2. ✅ **Sentry Application Pods** - Stateless, easy rollback
3. ✅ **Memcached** - Stateless cache, low impact
4. ⚠️ **RabbitMQ** - Stateful, moderate risk
5. ⛔ **ClickHouse** - Stateful, high risk, requires backup
6. ⛔ **Zookeeper** - Critical dependency, highest risk

**Recommended Timeline:**

- **Week 1:** Test all upgrades in ds-test cluster
- **Week 2:** Upgrade EKS Node OS + Sentry Apps + Memcached in production
- **Week 3:** Upgrade RabbitMQ in production
- **Week 4:** Upgrade ClickHouse in production (with maintenance window)
- **Week 5:** Upgrade Zookeeper in production (with maintenance window)

---

## 1. EKS Node OS (Amazon Linux 2 → AL2023)

**Priority:** 1  
**Complexity:** Low  
**Estimated Duration:** 45-90 minutes  
**Downtime Risk:** Minimal (if done correctly)

---

### Prerequisites

- [ ] Verify all pods have proper resource requests/limits set
- [ ] Ensure Pod Disruption Budgets (PDBs) are configured
- [ ] Backup critical data (ClickHouse, RabbitMQ, Zookeeper)
- [ ] Verify cluster has enough capacity for pod rescheduling
- [ ] Test in ds-test cluster first
- [ ] Schedule during low-traffic window

---

### Pre-Flight Checks

#### Check current node group configuration

```bash
aws eks describe-nodegroup --cluster-name sentry-xx-eks-skd5s-d94nn --nodegroup-name sentry-xx-eks-skd5s-kwlxd --region us-west-2
```

#### Verify pod distribution across nodes

```bash
kubectl get pods -n sentry -o wide
```

#### Check for PodDisruptionBudgets

```bash
kubectl get pdb -n sentry
```

#### Verify node health

```bash
kubectl get nodes -o wide
```

#### Check for any pending pods

```bash
kubectl get pods -n sentry --field-selector=status.phase=Pending
```

---

### Execution Steps

#### Step 1: Create PodDisruptionBudgets if not present

```bash
# For Sentry web pods
kubectl apply -f - <<EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: sentry-web-pdb
  namespace: sentry
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: sentry
      component: web
EOF
```

**⚠️ Important:** Create PDBs for all critical services to ensure availability during node drain

#### Step 2: Get latest AL2023 AMI ID for EKS 1.31

```bash
aws ssm get-parameter --name /aws/service/eks/optimized-ami/1.31/amazon-linux-2023/x86_64/standard/recommended/image_id --region us-west-2 --query 'Parameter.Value' --output text
```

**⚠️ Important:** This retrieves the latest EKS-optimized AL2023 AMI

#### Step 3: Update node group to AL2023

```bash
aws eks update-nodegroup-version \
  --cluster-name sentry-xx-eks-skd5s-d94nn \
  --nodegroup-name sentry-xx-eks-skd5s-kwlxd \
  --region us-west-2 \
  --force
```

**⚠️ Important:** EKS will perform rolling update. Monitor progress in AWS console or via CLI

#### Step 4: Monitor the update progress

```bash
# Watch node group status
watch -n 10 'aws eks describe-nodegroup --cluster-name sentry-xx-eks-skd5s-d94nn --nodegroup-name sentry-xx-eks-skd5s-kwlxd --region us-west-2 --query "nodegroup.status" --output text'

# Watch nodes being replaced
watch -n 5 'kubectl get nodes -o wide'

# Monitor pod status
watch -n 5 'kubectl get pods -n sentry'
```

**⚠️ Important:** Update typically takes 45-90 minutes. Nodes are replaced one at a time.

#### Step 5: Verify all nodes are on AL2023

```bash
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeInfo.osImage}{"\n"}{end}'
```

**⚠️ Important:** All nodes should show 'Amazon Linux 2023'

---

### Post-Upgrade Validation

#### ✅ All nodes are running AL2023

```bash
kubectl get nodes -o wide
```

#### ✅ All pods are running and healthy

```bash
kubectl get pods -n sentry | grep -v Running
```

#### ✅ No pods are pending or crashing

```bash
kubectl get pods -n sentry --field-selector=status.phase!=Running,status.phase!=Succeeded
```

#### ✅ Check Sentry application health

```bash
kubectl exec -n sentry deployment/sentry-web -- sentry health
```

---

### Rollback Plan

- If issues occur, EKS allows you to roll back to previous AMI
- Command: aws eks update-nodegroup-version --cluster-name <cluster> --nodegroup-name <nodegroup> --launch-template name=<old-template>,version=<old-version>
- Alternatively, scale up old node group and scale down new one


---

## 2. Sentry Application Pods (Web, Worker, Cron)

**Priority:** 2  
**Complexity:** Low-Medium  
**Estimated Duration:** 30-45 minutes  
**Downtime Risk:** None (rolling update)

---

### Prerequisites

- [ ] Node OS upgrade completed successfully
- [ ] Review Sentry changelog for breaking changes
- [ ] Backup Postgres database
- [ ] Test new version in ds-test cluster
- [ ] Notify team of upgrade window

---

### Pre-Flight Checks

#### Check current Sentry version

```bash
kubectl get deployment -n sentry -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}' | grep sentry
```

#### Check available updates

```bash
# Visit https://github.com/getsentry/self-hosted/releases
```

**Note:** Check for versions newer than 25.1.0

#### Verify current health

```bash
kubectl exec -n sentry deployment/sentry-web -- sentry health
```

---

### Execution Steps

#### Step 1: Backup Postgres database

```bash
# Get postgres pod name
POSTGRES_POD=$(kubectl get pod -n sentry -l app=postgresql -o jsonpath='{.items[0].metadata.name}')

# Create backup
kubectl exec -n sentry $POSTGRES_POD -- pg_dump -U sentry sentry > sentry-backup-$(date +%Y%m%d-%H%M%S).sql
```

**⚠️ Important:** Store backup in secure location

#### Step 2: Update Sentry image version (if using Helm)

```bash
# Get current Helm release
helm list -n sentry

# Update values file or use --set
helm upgrade sentry sentry/sentry \
  --namespace sentry \
  --set image.tag=25.2.0 \
  --reuse-values
```

**⚠️ Important:** Replace 25.2.0 with desired version

#### Step 3: Update Sentry image version (if using kubectl)

```bash
# Update web deployment
kubectl set image deployment/sentry-web sentry-web=getsentry/sentry:25.2.0 -n sentry

# Update worker deployment
kubectl set image deployment/sentry-worker sentry-worker=getsentry/sentry:25.2.0 -n sentry

# Update cron deployment
kubectl set image deployment/sentry-cron sentry-cron=getsentry/sentry:25.2.0 -n sentry
```

**⚠️ Important:** Kubernetes will perform rolling update automatically

#### Step 4: Monitor rollout status

```bash
kubectl rollout status deployment/sentry-web -n sentry
kubectl rollout status deployment/sentry-worker -n sentry
kubectl rollout status deployment/sentry-cron -n sentry
```

**⚠️ Important:** Wait for each rollout to complete successfully

#### Step 5: Run database migrations (if needed)

```bash
kubectl exec -n sentry deployment/sentry-web -- sentry upgrade --noinput
```

**⚠️ Important:** Sentry will apply any necessary schema changes

---

### Post-Upgrade Validation

#### ✅ Verify new version is running

```bash
kubectl get pods -n sentry -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}' | grep sentry
```

#### ✅ Check pod health

```bash
kubectl get pods -n sentry | grep sentry
```

#### ✅ Verify Sentry application health

```bash
kubectl exec -n sentry deployment/sentry-web -- sentry health
```

#### ✅ Test web UI access

```bash
# Access Sentry UI and verify functionality
```

#### ✅ Check for errors in logs

```bash
kubectl logs -n sentry deployment/sentry-web --tail=100 | grep -i error
```

---

### Rollback Plan

- kubectl rollout undo deployment/sentry-web -n sentry
- kubectl rollout undo deployment/sentry-worker -n sentry
- kubectl rollout undo deployment/sentry-cron -n sentry
- Or: helm rollback sentry -n sentry


---

## 3. Memcached

**Priority:** 3  
**Complexity:** Low  
**Estimated Duration:** 10-15 minutes  
**Downtime Risk:** Low (cache will be rebuilt)

---

### Prerequisites

- [ ] Verify current version: memcached:1.6.32-alpine
- [ ] Check for newer stable versions
- [ ] Understand that cache will be cleared during upgrade

---

### Pre-Flight Checks

---

### Execution Steps

#### Step 1: Update Memcached image

```bash
kubectl set image deployment/sentry-memcached memcached=memcached:1.6.33-alpine -n sentry
```

**⚠️ Important:** Cache will be cleared but will rebuild automatically

#### Step 2: Monitor rollout

```bash
kubectl rollout status deployment/sentry-memcached -n sentry
```

---

### Post-Upgrade Validation

#### ✅ Verify pods are running

```bash
kubectl get pods -n sentry -l app=memcached
```

#### ✅ Test connectivity

```bash
kubectl exec -n sentry deployment/sentry-memcached -- memcached-tool localhost:11211 stats
```

---

### Rollback Plan



---

## 4. RabbitMQ StatefulSet

**Priority:** 4  
**Complexity:** Medium  
**Estimated Duration:** 45-60 minutes  
**Downtime Risk:** Medium (brief interruptions possible)

---

### Prerequisites

- [ ] Current version: bitnami/rabbitmq:3.11.18-debian-11-r0
- [ ] Check RabbitMQ upgrade path (3.11.x → 3.12.x → 3.13.x)
- [ ] Backup RabbitMQ definitions and messages
- [ ] Ensure cluster has 3 replicas for HA
- [ ] Schedule during low-traffic window

---

### Pre-Flight Checks

#### Check RabbitMQ cluster status

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status
```

#### Check queue status

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl list_queues name messages consumers
```

#### Export definitions

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl export_definitions /tmp/definitions.json
```

---

### Execution Steps

#### Step 1: Backup RabbitMQ definitions

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl export_definitions /tmp/definitions-backup-$(date +%Y%m%d).json
kubectl cp sentry/sentry-rabbitmq-0:/tmp/definitions-backup-*.json ./rabbitmq-backup/
```

**⚠️ Important:** Keep backup for rollback purposes

#### Step 2: Update StatefulSet with new image (incremental upgrade to 3.12.x first)

```bash
kubectl patch statefulset sentry-rabbitmq -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"rabbitmq","image":"bitnami/rabbitmq:3.12.14-debian-12-r0"}]}}}}'
```

**⚠️ Important:** Don't skip major versions - go 3.11 → 3.12 → 3.13

#### Step 3: Upgrade pods one at a time

```bash
# Delete pod 0 and wait for it to be recreated
kubectl delete pod sentry-rabbitmq-0 -n sentry
kubectl wait --for=condition=ready pod/sentry-rabbitmq-0 -n sentry --timeout=300s

# Verify cluster health before proceeding
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status

# Delete pod 1
kubectl delete pod sentry-rabbitmq-1 -n sentry
kubectl wait --for=condition=ready pod/sentry-rabbitmq-1 -n sentry --timeout=300s

# Verify cluster health
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status

# Delete pod 2
kubectl delete pod sentry-rabbitmq-2 -n sentry
kubectl wait --for=condition=ready pod/sentry-rabbitmq-2 -n sentry --timeout=300s
```

**⚠️ Important:** CRITICAL: Only upgrade one pod at a time. Wait for cluster to stabilize between each pod.

#### Step 4: After 3.12.x is stable, upgrade to 3.13.x

```bash
# Wait 10-15 minutes to ensure 3.12 is stable
sleep 900

# Update to 3.13.x
kubectl patch statefulset sentry-rabbitmq -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"rabbitmq","image":"bitnami/rabbitmq:3.13.7-debian-12-r7"}]}}}}'

# Repeat one-by-one pod deletion as in step 3
```

**⚠️ Important:** Multi-stage upgrade reduces risk

---

### Post-Upgrade Validation

#### ✅ Verify all pods are running new version

```bash
kubectl get pods -n sentry -l app=rabbitmq -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'
```

#### ✅ Check cluster status

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status
```

#### ✅ Verify all nodes are running

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl list_queues
```

#### ✅ Check for alarms

```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl list_alarms
```

#### ✅ Verify Sentry can connect

```bash
kubectl logs -n sentry deployment/sentry-worker --tail=50 | grep -i rabbitmq
```

---

### Rollback Plan

- Rolling back RabbitMQ is complex and not recommended
- Better approach: restore from backup to a new StatefulSet
- Or: Scale down to 1 replica, restore definitions, scale back up
- Prevention: Test in ds-test cluster first


---

## 5. ClickHouse StatefulSet

**Priority:** 5  
**Complexity:** High  
**Estimated Duration:** 60-90 minutes  
**Downtime Risk:** Medium-High (read-only mode during upgrade)

---

### Prerequisites

- [ ] Current version: clickhouse/clickhouse-server:23.8.16.16
- [ ] Schedule maintenance window (Sentry will have degraded performance)
- [ ] Full backup of ClickHouse data
- [ ] Review ClickHouse changelog for breaking changes
- [ ] Test in ds-test cluster first
- [ ] Ensure adequate disk space for backup

---

### Pre-Flight Checks

#### Check ClickHouse cluster status

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.clusters'
```

#### Check replication status

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT database, table, is_leader, total_replicas, active_replicas FROM system.replicas'
```

#### Check disk usage

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- df -h /var/lib/clickhouse
```

#### Identify tables and sizes

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT database, table, formatReadableSize(sum(bytes)) as size FROM system.parts GROUP BY database, table ORDER BY sum(bytes) DESC'
```

---

### Execution Steps

#### Step 1: Create backup of ClickHouse data

```bash
# Create backup using ClickHouse backup tool or snapshots
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'BACKUP DATABASE default TO Disk(\'backups\', \'backup-$(date +%Y%m%d-%H%M%S).zip\')'

# Or use volume snapshots if using EBS
# Get PVC names
kubectl get pvc -n sentry -l app=clickhouse

# Create EBS snapshots via AWS CLI
# aws ec2 create-snapshot --volume-id <volume-id> --description 'ClickHouse backup before upgrade'
```

**⚠️ Important:** Backup is CRITICAL. ClickHouse upgrades can fail and corrupt data.

#### Step 2: Put Sentry in maintenance mode (optional but recommended)

```bash
# Scale down Sentry workers to reduce load
kubectl scale deployment sentry-worker -n sentry --replicas=1

# Or enable read-only mode in Sentry config
```

**⚠️ Important:** Reduces risk of data inconsistency during upgrade

#### Step 3: Upgrade ClickHouse incrementally (23.8 → 23.12 → 24.x)

```bash
# First upgrade to 23.12 LTS
kubectl patch statefulset sentry-clickhouse -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"clickhouse","image":"clickhouse/clickhouse-server:23.12.6.19"}]}}}}'
```

**⚠️ Important:** ClickHouse requires incremental upgrades between major versions

#### Step 4: Upgrade pod 0 first (leader)

```bash
# Delete pod 0
kubectl delete pod sentry-clickhouse-0 -n sentry

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/sentry-clickhouse-0 -n sentry --timeout=600s

# Verify it started successfully
kubectl logs -n sentry sentry-clickhouse-0 --tail=50

# Check cluster status
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT version()'
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.clusters'
```

**⚠️ Important:** Wait 5-10 minutes and monitor logs before proceeding

#### Step 5: Upgrade remaining pods one at a time

```bash
# Delete pod 1
kubectl delete pod sentry-clickhouse-1 -n sentry
kubectl wait --for=condition=ready pod/sentry-clickhouse-1 -n sentry --timeout=600s

# Verify replication is working
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.replicas WHERE active_replicas < total_replicas'

# Wait for replication to catch up (should return empty)
# Then proceed to pod 2
kubectl delete pod sentry-clickhouse-2 -n sentry
kubectl wait --for=condition=ready pod/sentry-clickhouse-2 -n sentry --timeout=600s
```

**⚠️ Important:** CRITICAL: Wait for replication to complete between each pod. Check system.replicas.

#### Step 6: After 23.12 is stable, upgrade to 24.x LTS

```bash
# Wait 15-20 minutes to ensure 23.12 is stable
sleep 1200

# Update to 24.x
kubectl patch statefulset sentry-clickhouse -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"clickhouse","image":"clickhouse/clickhouse-server:24.3.12.75"}]}}}}'

# Repeat one-by-one pod deletion as in steps 4-5
```

**⚠️ Important:** 24.3 is LTS version. Avoid .0 releases.

#### Step 7: Restore Sentry to normal operation

```bash
# Scale workers back up
kubectl scale deployment sentry-worker -n sentry --replicas=3
```

---

### Post-Upgrade Validation

#### ✅ Verify all pods running new version

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT hostname(), version() FROM cluster(\"default\", system.one)'
```

#### ✅ Check replication status

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT database, table, is_leader, total_replicas, active_replicas, log_pointer, absolute_delay FROM system.replicas'
```

#### ✅ Verify no replication delays

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.replicas WHERE absolute_delay > 0'
```

#### ✅ Check for errors in logs

```bash
kubectl logs -n sentry sentry-clickhouse-0 --tail=100 | grep -i error
```

#### ✅ Verify Sentry can query ClickHouse

```bash
kubectl exec -n sentry deployment/sentry-web -- python -c "from sentry.utils import snuba; print(snuba.query(...))"
```

#### ✅ Check system tables for issues

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.errors WHERE last_error_time > now() - INTERVAL 1 HOUR'
```

---

### Rollback Plan

- ClickHouse rollback is VERY difficult - prevention is key
- If upgrade fails on pod 0:
-   1. Delete pod 0: kubectl delete pod sentry-clickhouse-0 -n sentry
-   2. Revert StatefulSet image to old version
-   3. Pod 0 will recreate with old version
- If upgrade fails after multiple pods:
-   1. Stop all writes to ClickHouse
-   2. Delete all ClickHouse pods
-   3. Restore from backup/snapshot
-   4. Revert StatefulSet to old version
-   5. Recreate pods
- Best practice: Test in ds-test first, have backup ready

---

### ⚠️ Critical Warnings

⚠️ ClickHouse upgrades are HIGH RISK

⚠️ ALWAYS backup before upgrading

⚠️ Test in ds-test cluster first

⚠️ Schedule maintenance window - Sentry will have degraded performance

⚠️ Monitor replication status carefully between each pod

⚠️ Never upgrade more than one pod at a time

⚠️ Wait for replication to fully catch up before proceeding

⚠️ Keep old version backup for at least 7 days after upgrade


---

## 6. Zookeeper StatefulSet

**Priority:** 6  
**Complexity:** Very High  
**Estimated Duration:** 60-90 minutes  
**Downtime Risk:** High (ClickHouse depends on it)

---

### Prerequisites

- [ ] Current version: bitnamilegacy/zookeeper:3.8.2-debian-11-r27
- [ ] ClickHouse upgrade completed and stable
- [ ] Full backup of Zookeeper data
- [ ] Schedule maintenance window
- [ ] Understand that ClickHouse will be affected
- [ ] Test in ds-test cluster first
- [ ] Have rollback plan ready

---

### Pre-Flight Checks

#### Check Zookeeper ensemble status

```bash
kubectl exec -n sentry sentry-zookeeper-0 -- zkServer.sh status
```

#### Identify leader

```bash
for i in 0 1 2; do echo "Pod $i:"; kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status | grep Mode; done
```

#### Check Zookeeper data

```bash
kubectl exec -n sentry sentry-zookeeper-0 -- zkCli.sh ls /
```

#### Verify ClickHouse connection to Zookeeper

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.zookeeper WHERE path = \"/\"'
```

---

### Execution Steps

#### Step 1: Backup Zookeeper data

```bash
# Create snapshot of Zookeeper data
for i in 0 1 2; do
  kubectl exec -n sentry sentry-zookeeper-$i -- tar czf /tmp/zk-backup-$i-$(date +%Y%m%d).tar.gz /bitnami/zookeeper/data
  kubectl cp sentry/sentry-zookeeper-$i:/tmp/zk-backup-$i-*.tar.gz ./zk-backup/
done

# Or use volume snapshots for EBS-backed PVCs
```

**⚠️ Important:** Backup is CRITICAL. Zookeeper holds ClickHouse metadata.

#### Step 2: Put ClickHouse in read-only mode (optional)

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SYSTEM STOP MERGES'
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SYSTEM STOP FETCHES'
```

**⚠️ Important:** Reduces risk during Zookeeper upgrade

#### Step 3: Upgrade follower pods first (NOT the leader)

```bash
# Identify leader first
for i in 0 1 2; do echo "Pod $i:"; kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status | grep Mode; done

# Assume pod 0 is leader, upgrade pod 1 first
kubectl patch statefulset sentry-zookeeper -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"zookeeper","image":"bitnami/zookeeper:3.9.3-debian-12-r0"}]}}}}'

# Delete follower pod 1
kubectl delete pod sentry-zookeeper-1 -n sentry

# Wait for it to rejoin ensemble
kubectl wait --for=condition=ready pod/sentry-zookeeper-1 -n sentry --timeout=300s

# Verify it rejoined
kubectl exec -n sentry sentry-zookeeper-1 -- zkServer.sh status
```

**⚠️ Important:** NEVER upgrade leader first. Always upgrade followers first.

#### Step 4: Upgrade second follower

```bash
# Delete follower pod 2 (assuming it's also a follower)
kubectl delete pod sentry-zookeeper-2 -n sentry

# Wait for it to rejoin
kubectl wait --for=condition=ready pod/sentry-zookeeper-2 -n sentry --timeout=300s

# Verify ensemble health
kubectl exec -n sentry sentry-zookeeper-0 -- zkServer.sh status
kubectl exec -n sentry sentry-zookeeper-1 -- zkServer.sh status
kubectl exec -n sentry sentry-zookeeper-2 -- zkServer.sh status
```

**⚠️ Important:** Wait 5-10 minutes between each pod upgrade

#### Step 5: Upgrade leader last

```bash
# Delete leader pod (pod 0 in this example)
kubectl delete pod sentry-zookeeper-0 -n sentry

# Wait for it to rejoin
kubectl wait --for=condition=ready pod/sentry-zookeeper-0 -n sentry --timeout=300s

# Verify ensemble health and new leader election
for i in 0 1 2; do
  echo "Pod $i:"
  kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status
done
```

**⚠️ Important:** Leader election will occur automatically. New leader may be different pod.

#### Step 6: Restore ClickHouse to normal operation

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SYSTEM START MERGES'
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SYSTEM START FETCHES'
```

---

### Post-Upgrade Validation

#### ✅ Verify all Zookeeper pods are running new version

```bash
kubectl get pods -n sentry -l app=zookeeper -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'
```

#### ✅ Check ensemble status

```bash
for i in 0 1 2; do echo "Pod $i:"; kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status; done
```

#### ✅ Verify quorum is healthy

```bash
kubectl exec -n sentry sentry-zookeeper-0 -- zkCli.sh ls /
```

#### ✅ Verify ClickHouse can connect to Zookeeper

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.zookeeper WHERE path = \"/\"'
```

#### ✅ Check ClickHouse replication status

```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query 'SELECT * FROM system.replicas'
```

#### ✅ Verify no Zookeeper errors in ClickHouse logs

```bash
kubectl logs -n sentry sentry-clickhouse-0 --tail=100 | grep -i zookeeper
```

---

### Rollback Plan

- Zookeeper rollback is EXTREMELY difficult
- If upgrade fails:
-   1. Stop all ClickHouse writes immediately
-   2. Delete all Zookeeper pods
-   3. Revert StatefulSet to old image
-   4. Restore Zookeeper data from backup
-   5. Restart Zookeeper ensemble
-   6. Verify ClickHouse reconnects successfully
- Prevention is critical:
-   - Test in ds-test cluster first
-   - Have backup ready
-   - Schedule adequate maintenance window
-   - Monitor carefully during upgrade

---

### ⚠️ Critical Warnings

⚠️⚠️⚠️ HIGHEST RISK UPGRADE ⚠️⚠️⚠️

⚠️ Zookeeper is critical for ClickHouse operation

⚠️ Loss of quorum will take down ClickHouse

⚠️ ALWAYS upgrade followers before leader

⚠️ NEVER upgrade more than one pod at a time

⚠️ Wait for each pod to fully rejoin before proceeding

⚠️ Monitor ClickHouse status throughout the upgrade

⚠️ Have backup and rollback plan ready

⚠️ Test in ds-test cluster first - NO EXCEPTIONS

⚠️ Consider scheduling during off-hours with team on standby


---

## Post-Upgrade Validation

After completing all upgrades, perform these final checks:

### 1. Verify All Components

```bash
# Check all pods are running
kubectl get pods -n sentry

# Verify no pods are crashing
kubectl get pods -n sentry --field-selector=status.phase!=Running,status.phase!=Succeeded

# Check node versions
kubectl get nodes -o wide
```

### 2. Verify Sentry Application Health

```bash
# Check Sentry health endpoint
kubectl exec -n sentry deployment/sentry-web -- sentry health

# Test web UI access
# Open Sentry URL in browser and verify functionality

# Check for errors in logs
kubectl logs -n sentry deployment/sentry-web --tail=100 | grep -i error
kubectl logs -n sentry deployment/sentry-worker --tail=100 | grep -i error
```

### 3. Verify Data Pipeline

```bash
# Check ClickHouse is receiving data
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query "SELECT count() FROM default.sentry_local WHERE timestamp > now() - INTERVAL 5 MINUTE"

# Check RabbitMQ queues
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl list_queues name messages

# Verify Zookeeper ensemble
for i in 0 1 2; do 
  echo "Zookeeper pod $i:"
  kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status
done
```

### 4. Performance Testing

```bash
# Send test events to Sentry
# Monitor processing time
# Check for any performance degradation
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Pod stuck in Pending state

**Symptoms:**
```bash
kubectl get pods -n sentry
# Shows pod in Pending state
```

**Solution:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n sentry

# Common causes:
# 1. Insufficient resources - scale up node group
# 2. PVC not binding - check storage class
# 3. Node selector mismatch - check pod spec
```

---

#### Issue: ClickHouse replication lag

**Symptoms:**
```bash
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query "SELECT * FROM system.replicas WHERE absolute_delay > 0"
# Shows replication delay
```

**Solution:**
```bash
# Check network connectivity between pods
kubectl exec -n sentry sentry-clickhouse-0 -- ping sentry-clickhouse-1.sentry-clickhouse.sentry.svc.cluster.local

# Check ClickHouse logs
kubectl logs -n sentry sentry-clickhouse-0 --tail=200 | grep -i replication

# If delay is large, may need to restart replication
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query "SYSTEM RESTART REPLICA default.table_name"
```

---

#### Issue: RabbitMQ cluster split-brain

**Symptoms:**
```bash
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status
# Shows nodes not in sync
```

**Solution:**
```bash
# Stop the minority partition nodes
kubectl exec -n sentry sentry-rabbitmq-1 -- rabbitmqctl stop_app

# Reset the node
kubectl exec -n sentry sentry-rabbitmq-1 -- rabbitmqctl reset

# Rejoin cluster
kubectl exec -n sentry sentry-rabbitmq-1 -- rabbitmqctl join_cluster rabbit@sentry-rabbitmq-0
kubectl exec -n sentry sentry-rabbitmq-1 -- rabbitmqctl start_app

# Verify cluster status
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl cluster_status
```

---

#### Issue: Zookeeper lost quorum

**Symptoms:**
```bash
kubectl exec -n sentry sentry-zookeeper-0 -- zkServer.sh status
# Shows "Mode: standalone" or connection refused
```

**Solution:**
```bash
# This is CRITICAL - ClickHouse will be down

# Check all Zookeeper pods
for i in 0 1 2; do
  echo "Pod $i:"
  kubectl exec -n sentry sentry-zookeeper-$i -- zkServer.sh status
done

# If quorum is lost, restart pods one at a time
kubectl delete pod sentry-zookeeper-0 -n sentry
kubectl wait --for=condition=ready pod/sentry-zookeeper-0 -n sentry --timeout=300s

# Verify quorum restored
kubectl exec -n sentry sentry-zookeeper-0 -- zkServer.sh status

# If still failing, may need to restore from backup (see Zookeeper rollback plan)
```

---

#### Issue: Node drain stuck during EKS upgrade

**Symptoms:**
- EKS node group update stuck at "Draining nodes"
- Pods not evicting from old nodes

**Solution:**
```bash
# Identify pods blocking drain
kubectl get pods -n sentry -o wide | grep <old-node-name>

# Check for pods without PodDisruptionBudget
kubectl get pdb -n sentry

# Force delete stuck pods (last resort)
kubectl delete pod <pod-name> -n sentry --force --grace-period=0

# Or cordon and drain manually
kubectl cordon <node-name>
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force
```

---

#### Issue: Sentry web UI showing errors after upgrade

**Symptoms:**
- 500 errors in web UI
- "Database migration required" message

**Solution:**
```bash
# Run database migrations
kubectl exec -n sentry deployment/sentry-web -- sentry upgrade --noinput

# Clear cache
kubectl exec -n sentry deployment/sentry-memcached -- echo "flush_all" | nc localhost 11211

# Restart web pods
kubectl rollout restart deployment/sentry-web -n sentry

# Check logs for specific errors
kubectl logs -n sentry deployment/sentry-web --tail=200
```

---

## Emergency Contacts and Resources

### Team Contacts
- **Primary:** [Your Name] - [Contact Info]
- **Backup:** [Backup Person] - [Contact Info]
- **On-Call:** [On-Call Contact]

### Useful Resources
- Sentry Self-Hosted Docs: https://develop.sentry.dev/self-hosted/
- ClickHouse Docs: https://clickhouse.com/docs
- RabbitMQ Clustering: https://www.rabbitmq.com/clustering.html
- Zookeeper Admin: https://zookeeper.apache.org/doc/current/zookeeperAdmin.html
- EKS Best Practices: https://aws.github.io/aws-eks-best-practices/

### Monitoring Dashboards
- Sentry UI: [Your Sentry URL]
- Grafana: [Your Grafana URL]
- CloudWatch: [AWS Console Link]

---

## Appendix: Quick Reference Commands

### Check All Component Versions

```bash
# Nodes
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"	"}{.status.nodeInfo.osImage}{"
"}{end}'

# Sentry
kubectl get deployment -n sentry -o jsonpath='{range .items[*]}{.metadata.name}{"	"}{.spec.template.spec.containers[0].image}{"
"}{end}'

# StatefulSets
kubectl get statefulset -n sentry -o jsonpath='{range .items[*]}{.metadata.name}{"	"}{.spec.template.spec.containers[0].image}{"
"}{end}'
```

### Quick Health Checks

```bash
# All pods status
kubectl get pods -n sentry

# ClickHouse health
kubectl exec -n sentry sentry-clickhouse-0 -- clickhouse-client --query "SELECT 1"

# RabbitMQ health
kubectl exec -n sentry sentry-rabbitmq-0 -- rabbitmqctl status

# Zookeeper health
kubectl exec -n sentry sentry-zookeeper-0 -- zkServer.sh status

# Sentry health
kubectl exec -n sentry deployment/sentry-web -- sentry health
```

### Emergency Rollback Commands

```bash
# Rollback Kubernetes deployment
kubectl rollout undo deployment/<deployment-name> -n sentry

# Rollback Helm release
helm rollback <release-name> -n sentry

# Rollback StatefulSet (requires manual intervention)
kubectl patch statefulset <statefulset-name> -n sentry -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","image":"<old-image>"}]}}}}'
kubectl delete pod <pod-name> -n sentry  # Repeat for each pod
```

---

## Sign-Off Checklist

After completing all upgrades:

- [ ] All components upgraded to target versions
- [ ] All pods running and healthy
- [ ] No errors in application logs
- [ ] Sentry web UI accessible and functional
- [ ] Test events successfully processed
- [ ] ClickHouse replication healthy
- [ ] RabbitMQ cluster healthy
- [ ] Zookeeper ensemble healthy
- [ ] Performance metrics normal
- [ ] Monitoring and alerting functional
- [ ] Backups verified and stored securely
- [ ] Documentation updated with new versions
- [ ] Team notified of completion
- [ ] Post-mortem scheduled (if issues occurred)

---

**Document Version:** 1.0  
**Last Updated:** {plan['metadata']['created_date']}  
**Next Review:** [Schedule quarterly review]

