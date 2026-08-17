# Dish-Chat Deployment Guide
## Transfer from 3080 to 3090

**Source:** montjac@10.79.85.47:~/dish-chat  
**Target:** jakebot@10.79.85.35:~/Jakes-agent  
**Date:** 2026-02-27

---

## Overview

This guide walks through the complete process of transferring the Dish-Chat application from the 3080 development server to a clean installation on the 3090 server under a dedicated `jakebot` user.

---

## Prerequisites on Target System (3090)

Before starting, ensure the following are installed on jakebot@10.79.85.35:

### Required Software
- **Python 3.12** with venv and dev packages
  ```bash
  sudo apt-get update
  sudo apt-get install python3.12 python3.12-venv python3.12-dev
  ```

- **Docker** and Docker Compose
  ```bash
  # Follow official Docker installation guide
  # https://docs.docker.com/engine/install/ubuntu/
  ```

- **AWS CLI and SecGateway** (for AWS token refresh)
  - SecGateway should be installed at: `~/secgateway/bin/secgateway.py`

### System Access
- SSH access to both 3080 (source) and 3090 (target)
- Docker permissions for jakebot user:
  ```bash
  sudo usermod -aG docker jakebot
  ```

---

## Deployment Steps

### Step 1: Transfer Files from 3080 to 3090

From a machine with SSH access to both servers (or from the 3080):

```bash
# Make script executable
chmod +x transfer-dishchat.sh

# Run transfer script
./transfer-dishchat.sh
```

**What this does:**
- Creates `/home/jakebot/Jakes-agent` directory on 3090
- Transfers all required application files via rsync
- Excludes unnecessary files:
  - Python cache (__pycache__, *.pyc)
  - Virtual environment (.venv)
  - Git repository (.git)
  - Log files and backups
  - Temporary files

**Expected time:** 2-5 minutes depending on network speed

---

### Step 2: Setup on Target System (3090)

SSH into the target system:
```bash
ssh jakebot@10.79.85.35
cd ~/Jakes-agent
```

Copy the setup scripts to the target directory:
```bash
# These files should be in ~/Jakes-agent after transfer:
# - setup-dishchat.sh
# - start-dishchat.sh
# - stop-dishchat.sh
# - restart-dishchat.sh
```

Run the setup script:
```bash
bash setup-dishchat.sh
```

**What this does:**
1. Checks system prerequisites (Python 3.12, Docker, Docker Compose)
2. Creates Python virtual environment with Python 3.12
3. Installs Python dependencies from requirements.txt
4. Starts PostgreSQL container with pgvector extension
5. Runs Alembic database migrations
6. Creates log directory
7. Sets up .env configuration file
8. Verifies the installation

**Expected time:** 5-10 minutes

---

### Step 3: Start Dish-Chat

```bash
bash start-dishchat.sh
```

**What this does:**
1. Refreshes AWS tokens via SecGateway
2. Verifies PostgreSQL is running
3. Starts the FastAPI backend on port 8000
4. Saves PID to backend.pid
5. Verifies the backend is responding

**Expected time:** 10-30 seconds

---

### Step 4: Verify Deployment

#### Check Backend Status
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

#### Check Logs
```bash
tail -f ~/Jakes-agent/logs/backend.log
```

#### Check PostgreSQL
```bash
docker ps | grep postgres-dev-dishchat
docker logs postgres-dev-dishchat
```

#### Test API Endpoints
```bash
# Health check
curl http://10.79.85.35:8000/health

# API docs (open in browser)
http://10.79.85.35:8000/docs
```

---

## Service Management

### Start Dish-Chat
```bash
cd ~/Jakes-agent
bash start-dishchat.sh
```

### Stop Dish-Chat
```bash
cd ~/Jakes-agent
bash stop-dishchat.sh
```

### Restart Dish-Chat
```bash
cd ~/Jakes-agent
bash restart-dishchat.sh
```

### View Logs
```bash
# Real-time logs
tail -f ~/Jakes-agent/logs/backend.log

# Last 100 lines
tail -n 100 ~/Jakes-agent/logs/backend.log
```

---

## Configuration Files

### .env File
Located at: `~/Jakes-agent/.env`

Default configuration:
```bash
# Sentry Configuration
SENTRY_AUTH_TOKEN=<INJECT_FROM_SECRET_MANAGER>
SENTRY_ORG=dishtv.technology
SENTRY_URL=https://ds-testing-sentry

# AI Thought Visualization Configuration
AGENT_THOUGHT_CAPTURE_ENABLED=true
AGENT_VIZ_SERVER_URL=http://localhost:8000/rest/api/v1/viz/event
FASTAPI_PORT=8000
FASTAPI_HOST=0.0.0.0
```

### app/config.py
Database configuration:
- **Host:** 127.0.0.1
- **Port:** 5433 (mapped from container's 5432)
- **Database:** dishchat
- **User:** dev_user
- **Password:** dev123

---

## Architecture

### Components

1. **FastAPI Backend** (Python 3.12)
   - Port: 8000
   - Location: ~/Jakes-agent/app/
   - Virtual env: ~/Jakes-agent/.venv/

2. **PostgreSQL Database** (Docker)
   - Container: postgres-dev-dishchat
   - Image: pgvector/pgvector:pg17
   - Port: 5433 (host) -> 5432 (container)
   - Volume: pgdata_dishchat

3. **AWS Bedrock Integration**
   - Model: claude-sonnet-4-5
   - Region: us-east-1
   - Requires: SecGateway token refresh

### Directory Structure
```
~/Jakes-agent/
├── app/                      # Main application code
│   ├── agent/               # Agent logic and tools
│   ├── alembic/             # Database migrations
│   ├── chat/                # Chat functionality
│   ├── main.py              # FastAPI entry point
│   └── config.py            # Configuration
├── dev_postgres/            # PostgreSQL Docker setup
│   └── docker-compose.yaml
├── logs/                    # Application logs
├── .venv/                   # Python virtual environment
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── backend.pid              # Backend process ID
├── setup-dishchat.sh        # Setup script
├── start-dishchat.sh        # Start script
├── stop-dishchat.sh         # Stop script
└── restart-dishchat.sh      # Restart script
```

---

## Troubleshooting

### Backend Won't Start

**Check logs:**
```bash
tail -f ~/Jakes-agent/logs/backend.log
```

**Common issues:**
- PostgreSQL not running: `cd ~/Jakes-agent/dev_postgres && docker compose up -d`
- Port 8000 already in use: Check for conflicting processes
- Missing dependencies: Re-run `pip install -r requirements.txt`

### PostgreSQL Connection Issues

**Check container status:**
```bash
docker ps | grep postgres-dev-dishchat
docker logs postgres-dev-dishchat
```

**Restart PostgreSQL:**
```bash
cd ~/Jakes-agent/dev_postgres
docker compose down
docker compose up -d
```

### AWS Token Issues

**Refresh tokens manually:**
```bash
python3 ~/secgateway/bin/secgateway.py
```

**Check AWS credentials:**
```bash
aws sts get-caller-identity
```

### Database Migration Issues

**Check migration status:**
```bash
cd ~/Jakes-agent/app
source ../.venv/bin/activate
alembic current
alembic history
```

**Re-run migrations:**
```bash
alembic upgrade head
```

---

## Maintenance

### Update Dependencies
```bash
cd ~/Jakes-agent
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Database Backup
```bash
docker exec postgres-dev-dishchat pg_dump -U dev_user dishchat > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Database Restore
```bash
docker exec -i postgres-dev-dishchat psql -U dev_user dishchat < backup_file.sql
```

### Clean Restart
```bash
# Stop everything
bash stop-dishchat.sh

# Remove old logs
rm -f ~/Jakes-agent/logs/*

# Restart
bash start-dishchat.sh
```

---

## Key Differences from 3080

| Aspect | 3080 (Source) | 3090 (Target) |
|--------|---------------|---------------|
| User | montjac | jakebot |
| Path | /home/montjac/dish-chat | /home/jakebot/Jakes-agent |
| Virtual Env | Shared with JAMbot | Dedicated .venv |
| Database | Existing data | Fresh install |
| Git | .git included | .git excluded |
| Logs | Historical logs | Clean start |

---

## Support & Contacts

- **Repository:** (Add your Git repo URL here)
- **Documentation:** ~/Jakes-agent/README.md
- **Issues:** (Add your issue tracker URL here)

---

## Security Notes

- Ensure `.env` file permissions are restrictive: `chmod 600 .env`
- Keep AWS tokens refreshed via SecGateway
- PostgreSQL is exposed on localhost only (127.0.0.1:5433)
- Backend binds to 0.0.0.0:8000 - ensure firewall rules are appropriate

---

## Next Steps

After successful deployment:

1. **Test all endpoints** using the API documentation at http://10.79.85.35:8000/docs
2. **Configure monitoring** (if applicable)
3. **Set up automated backups** for PostgreSQL
4. **Document any custom configurations** specific to your environment
5. **Update DNS/load balancer** entries if this is a production deployment

---

*Last updated: 2026-02-27*

