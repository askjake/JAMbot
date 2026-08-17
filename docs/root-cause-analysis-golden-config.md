# Data Solutions Handoff — Root Cause Analysis

## App Overview
The Root Cause Analysis (RCA) app is a pipeline of 5 microservices. Each service needs to run in its own container inside the Kubernetes cluster. `api-service` is the only public-facing service and must be the ingress entry point. All other services are internal only.

## 1. Service Inventory

| Service | Port | Responsibility | Calls |
|---------|------|----------------|-------|
| api-service | 8000 | Receives external requests, validates them, triggers the pipeline | → log-fetcher |
| log-fetcher | 8001 | Fetches logs from the GHUH API and uploads them to S3 | → log-preprocessor |
| log-preprocessor | 8002 | Downloads logs from S3, extracts, concatenates, preprocesses, re-uploads to S3 | → log-analyzer |
| log-analyzer | 8003 | Reads preprocessed logs from S3, runs analysis via AWS Bedrock, saves reports to S3 | → analysis-report-formatter |
| analysis-report-formatter | 8004 | Fetches analysis reports from S3, generates PDFs, sends email via AWS SES | — |

## 2. Health Check Endpoints
All 5 services expose a `/health` endpoint with GET method returning `{"status": "healthy"}`. These can be used for liveness and readiness probes.

## 3. Environment Variables Per Service
Variables marked **(secret)** must be stored in a Kubernetes Secret, not a ConfigMap. All others go in a ConfigMap.

### api-service

| Variable | Description | Default |
|----------|-------------|---------|
| LOG_FETCHER_URL | Internal URL to log-fetcher | http://log-fetcher:8001/fetch-logs |
| LOG_FETCHER_TIMEOUT | Timeout in seconds for log-fetcher calls | 30 |

### log-fetcher

| Variable | Description | Default |
|----------|-------------|---------|
| GRASSHOPPER_API_URL | External GHUH API endpoint | https://grasshopperautoupload.dishanywhere.com:8443/grasshoppersmp/rest/v2/request/upload |
| GRASSHOPPER_AUTH_KEY | Auth key for GHUH API | **(secret — no default, pod fails at startup if missing)** |
| GRASSHOPPER_USERNAME | Username for GHUH API | HotPursuitRCA |
| UPLOAD_DESTINATION | Root URL GHUH uploads logs to | https://dsghuh.dishtv.technology/upload/ |
| S3_LOGS_LOCATION | S3 path where logs land after GHUH upload | s3://ds-ghuh-logs/ccshare/ |
| LOG_PREPROCESSOR_URL | Internal URL to log-preprocessor | http://log-preprocessor:8002/preprocess-logs |
| REPORT_FORMATTER_URL | Internal base URL to analysis-report-formatter (used by error handler) | http://analysis-report-formatter:8004 |
| SQS_QUEUE_URL | SQS queue URL for S3 upload notifications from GHUH | https://sqs.us-east-1.amazonaws.com/233532778289/RCA-GHUH |
| AWS_REGION_US_EAST_1 | AWS region for the SQS client | us-east-1 |
| SILENCE_DURATION_FOR_SQS | Minutes of silence (no new S3 files) before triggering preprocessor | 5 |
| TOTAL_CHECKS_BEFORE_TRIGGER_LOG_PRE_PROCESSOR | Number of consecutive silence intervals required before triggering | 3 |

### log-preprocessor

| Variable | Description | Default |
|----------|-------------|---------|
| LOG_ANALYZER_URL | Internal URL to log-analyzer | http://log-analyzer:8003/analyze-logs |
| S3_DESTINATION | S3 bucket for uploading preprocessed logs | s3://hot-pursuit-log-analysis/ |
| S3_CCSHARE_REGION | AWS region for the source S3 bucket (ccshare logs) | us-east-1 |

### log-analyzer

| Variable | Description | Default |
|----------|-------------|---------|
| BEDROCK_INFERENCE_PROFILE_ARN | ARN for the primary Bedrock inference profile (Claude Opus 4.5) | arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/p30i9173cxce |
| BEDROCK_COMBINATION_INFERENCE_PROFILE_ARN | ARN for the combination/summary Bedrock inference profile (Claude Haiku 4.5) | arn:aws:bedrock:us-west-2:233532778289:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0 |
| REQUEST_TIMEOUT | Timeout in seconds for downstream calls | 300 |

### analysis-report-formatter

| Variable | Description | Default |
|----------|-------------|---------|
| AWS_REGION_US_WEST_2 | AWS region for SES | us-west-2 |
| SES_FROM_EMAIL | Verified SES sender address | **(no default — must be set explicitly)** |

## 4. AWS Permissions Required Per Service (IRSA)
Each service pod needs an IAM role attached via IRSA. Below are the minimum permissions required per service.

### log-fetcher (S3 and SQS)
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::ds-ghuh-logs",
    "arn:aws:s3:::ds-ghuh-logs/*"
  ]
},
{
  "Effect": "Allow",
  "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
  "Resource": "arn:aws:sqs:us-east-1:233532778289:RCA-GHUH"
}
```

### log-preprocessor (S3)
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
  "Resource": [
    "arn:aws:s3:::ds-ghuh-logs",
    "arn:aws:s3:::ds-ghuh-logs/*",
    "arn:aws:s3:::hot-pursuit-log-analysis",
    "arn:aws:s3:::hot-pursuit-log-analysis/*"
  ]
}
```

### log-analyzer (S3 and AWS Bedrock)
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::hot-pursuit-log-analysis",
    "arn:aws:s3:::hot-pursuit-log-analysis/*"
  ]
},
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
  "Resource": [
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/p30i9173cxce",
    "arn:aws:bedrock:us-west-2:233532778289:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
  ]
}
```

### analysis-report-formatter (S3 and AWS SES)
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": [
    "arn:aws:s3:::hot-pursuit-log-analysis",
    "arn:aws:s3:::hot-pursuit-log-analysis/*"
  ]
},
{
  "Effect": "Allow",
  "Action": ["ses:SendEmail", "ses:SendRawMessage"],
  "Resource": "*"
}
```

## 5. ECR Repositories
All 5 repositories already exist under account 233532778289. The CI pipeline (`demoapp/.gitlab-ci.yml`) already builds and pushes to them on every merge to main. Only services which saw changes are built and pushed to ECR.

| Repository Name | Full URI |
|----------------|----------|
| root-cause-analysis/api-service | 233532778289.dkr.ecr.us-west-2.amazonaws.com/root-cause-analysis/api-service |
| root-cause-analysis/log-fetcher | 233532778289.dkr.ecr.us-west-2.amazonaws.com/root-cause-analysis/log-fetcher |
| root-cause-analysis/log-preprocessor | 233532778289.dkr.ecr.us-west-2.amazonaws.com/root-cause-analysis/log-preprocessor |
| root-cause-analysis/log-analyzer | 233532778289.dkr.ecr.us-west-2.amazonaws.com/root-cause-analysis/log-analyzer |
| root-cause-analysis/analysis-report-formatter | 233532778289.dkr.ecr.us-west-2.amazonaws.com/root-cause-analysis/analysis-report-formatter |

### Image Tagging Strategy (from .gitlab-ci.yml)
- `{VERSION}-{COMMIT_SHORT_SHA}` — primary tag, unique per build (e.g. `1.0.0-a1b2c3d`)
- `{VERSION}` — floating tag for the current version (e.g. `1.0.0`)
- `{ENVIRONMENT}` — floating tag for the current environment (currently `beta`)
- `latest` — always points to the most recent build

For the initial deployment, use the `latest` tag or the most recent `{VERSION}-{COMMIT_SHORT_SHA}` tag.

## 6. Ingress / Domain
- `api-service` is the only service that should be publicly accessible
- Domain Example: `root-cause-analysis.dishtv.technology`
- All traffic enters via `api-service` on port 8000
- The remaining 4 services should only be reachable internally within the cluster

## 7. Inter-Service Communication
Services communicate over HTTP using internal Kubernetes DNS. The expected URLs are:

| From | To | URL |
|------|-----|-----|
| api-service | log-fetcher | http://log-fetcher:8001/fetch-logs |
| log-fetcher | log-preprocessor | http://log-preprocessor:8002/preprocess-logs |
| log-preprocessor | log-analyzer | http://log-analyzer:8003/analyze-logs |
| log-analyzer | analysis-report-formatter | http://analysis-report-formatter:8004/process-analysis-reports |
| Any service (on critical error) | analysis-report-formatter | http://analysis-report-formatter:8004/send-error-email |

## 8. DNS and Load Balancer Provisioning

**Do not manually create Route53 records or ALB resources in AWS.**

All DNS and load balancer provisioning is fully automated:

- The **AWS Load Balancer Controller** watches for `Ingress` objects with `ingressClassName: alb` and provisions the ALB automatically from the annotations on the ingress manifest.
- **external-dns** watches for `Ingress` objects with the `external-dns.alpha.kubernetes.io/hostname` annotation and creates/updates the Route53 record automatically.

There is nothing to create manually. Define the hostname once in the Helm values (or ingress manifest), commit it to the repo, and the controllers handle the rest end-to-end.

```yaml
# Correct pattern — controllers do everything from these annotations alone
annotations:
  external-dns.alpha.kubernetes.io/hostname: "your-service.dishtv.technology"
  alb.ingress.kubernetes.io/scheme: internet-facing
  alb.ingress.kubernetes.io/target-type: ip
  alb.ingress.kubernetes.io/certificate-arn: "<acm-cert-arn>"
  alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
ingressClassName: alb
```

Creating records or load balancers manually in the AWS console or via CLI bypasses this automation, creates untracked resources, and risks conflicts with the controllers on next reconcile.

---

**Golden Configuration Reference**: This document serves as the authoritative specification for deploying microservice pipelines in Kubernetes with AWS service integration (S3, SQS, Bedrock, SES). Follow this pattern for similar data processing pipelines requiring:
- Multi-service orchestration
- AWS IAM role service account (IRSA) configuration
- Internal vs external service segmentation
- Health check and observability patterns
- CI/CD with ECR image management
