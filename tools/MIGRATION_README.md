# DynamoDB to PostgreSQL Migration Tool

Migrates chat data from DynamoDB to PostgreSQL, preserving conversation history and usage tracking.

## Prerequisites

**Environment Variables:**
```bash
export PAGE_TABLE="your-dynamodb-page-table"
export MESSAGE_TABLE="your-dynamodb-message-table" 
export USAGE_TABLE="your-dynamodb-usage-table"
export MASTER_KEY="your-base64-encryption-key"
```

**Database:** PostgreSQL running with migrations applied (`alembic upgrade head`)

## Usage

```bash
# Migrate all users
python migrate_dynamodb_to_postgres.py

# Migrate specific users
python migrate_dynamodb_to_postgres.py --users user1@dish.com user2@dish.com

# Debug mode (uses test user)
python migrate_dynamodb_to_postgres.py --debug --users user1@dish.com
```

## Data Flow

**Source (DynamoDB):** PAGE_TABLE → MESSAGE_TABLE → USAGE_TABLE  
**Target (PostgreSQL):** chat → message_metadata → usage_tracking → checkpoints

## Process

1. **Extract** - Queries DynamoDB, decrypts content, groups by user/chat
2. **Transform** - Creates Chat records, migrates UsageTracking with costs
3. **Load** - Simulates conversations via LangGraph, creates MessageMD records

## Limitations

- Skips vault-encrypted chats
- No rollback mechanism
- Requires all environment variables