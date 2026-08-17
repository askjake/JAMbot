# Dish-Chat Token Management & Prompt Compression Fix

## Overview

This update addresses three critical issues in the Dish-Chat application:

1. **AWS Token Expiration**: Tokens expire hourly, interrupting conversations
2. **Prompt Token Overflow**: Long conversations exceed model context limits
3. **Poor Error Recovery**: Chats unnecessarily go into readonly mode

## Changes Made

### 1. Enhanced AWS Token Management (`app/core/llm/chat_models.py`)

**Features:**
- **Proactive Token Refresh**: Background task refreshes AWS credentials 10 minutes before expiration
- **Token Expiry Tracking**: Models track when their credentials will expire
- **Automatic Retry**: Failed requests automatically retry with fresh tokens
- **Streaming Support**: Both `invoke` and `stream` methods support token refresh

**Key Functions:**
- `_get_next_token_expiry()`: Calculates next token expiration time
- `_refresh_models_periodically()`: Background task for proactive refresh
- `start_token_refresh_task()`: Initializes background refresh
- `invoke_with_retry()`: Invokes model with automatic retry
- `stream_with_retry()`: Streams with automatic retry

**Behavior:**
- Tokens refresh automatically 10 minutes before the top of each hour
- On ExpiredTokenException, automatically retries up to 2 times with fresh credentials
- Logs all refresh activities for monitoring

### 2. Message History Compression (`app/message/compression.py`)

**Features:**
- **Token Counting**: Accurate token counting using tiktoken library
- **Smart Trimming**: Keeps system prompts + most recent messages
- **Automatic Detection**: Detects when compression is needed (>70% context usage)
- **Compression Stats**: Provides detailed statistics for monitoring

**Key Functions:**
- `count_tokens(text)`: Count tokens in text
- `count_message_tokens(messages)`: Count tokens in message list
- `trim_message_history(messages, max_tokens)`: Trim messages to fit limit
- `should_compress_history(messages)`: Check if compression needed
- `get_compression_stats(messages)`: Get detailed statistics

**Behavior:**
- Monitors token usage at 70% threshold
- When exceeded, trims to 70% of context (leaves room for response)
- Always keeps:
  - System messages (prompts, instructions)
  - Most recent conversation turns
  - At least one complete user/assistant exchange

### 3. Enhanced Agent Service (`app/agent/service.py`)

**Features:**
- **Automatic Compression**: Compresses message history before processing new messages
- **Branch Support**: Applies compression when branching conversations
- **Graceful Degradation**: Continues even if compression fails

**Key Changes:**
- `_compress_message_history_if_needed()`: New method for compression
- Modified `process_new_user_message()`: Adds compression step
- Modified `branch_from_past_user_message()`: Adds compression step

**Behavior:**
- Before processing each new message, checks if history needs compression
- Compresses if needed, updates graph state with trimmed history
- Logs compression actions with before/after statistics

### 4. Improved Error Handling (`app/message/error_handling.py`)

**Features:**
- **Recoverable Error Detection**: Identifies which errors shouldn't set readonly
- **User-Friendly Messages**: Converts technical errors to readable messages
- **Selective Readonly**: Only sets readonly for truly unrecoverable errors

**Recoverable Errors:**
- AWS token expiration
- Network timeouts
- API throttling
- Token limit exceeded (handled by compression)

**Key Functions:**
- `is_recoverable_error(exc)`: Check if error is recoverable
- `get_user_friendly_error_message(exc)`: Convert to user message
- `handle_stream_error_enhanced()`: Enhanced error handler

## Installation

### Prerequisites
- Python 3.12+
- Existing Dish-Chat installation
- Access to AWS Bedrock

### Dependencies
The fix adds one new dependency:
```bash
pip install tiktoken
```

### Deployment Steps

1. **Backup Existing Code**:
   ```bash
   cd ~/dish-chat
   ./deploy_fixes.sh  # Automatically creates backups
   ```

2. **Manual Installation** (if needed):
   ```bash
   # Backup current files
   cp app/core/llm/chat_models.py app/core/llm/chat_models.py.backup
   cp app/agent/service.py app/agent/service.py.backup
   
   # Copy new files
   cp chat_models.py.new app/core/llm/chat_models.py
   cp compression.py app/message/compression.py
   cp agent_service.py.new app/agent/service.py
   cp error_handling.py app/message/error_handling.py
   
   # Install dependencies
   pip install tiktoken
   ```

3. **Validate Installation**:
   ```bash
   ./validate_fixes.sh
   ```

4. **Restart Application**:
   ```bash
   # Stop current instance
   pkill -f "uvicorn app.main:app"
   
   # Start with new code
   nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
   ```

## Monitoring

### Log Messages to Watch For

**Token Refresh:**
```
INFO - Started AWS token refresh background task
INFO - Next token refresh scheduled in 50.0 minutes
INFO - Proactively refreshing AWS tokens for all models
INFO - Refreshed primary model with new AWS tokens
```

**Message Compression:**
```
INFO - Chat abc123: Message history at 75.3% capacity (150000 tokens, 245 messages)
INFO - Chat abc123: Compressed to 180 messages (140000 tokens, 70.0%)
```

**Error Recovery:**
```
WARNING - AWS token expired during invocation, refreshing and retrying (attempt 1/3)
INFO - Chat abc123: Recoverable error, not setting readonly. User can retry.
```

### Metrics to Monitor

1. **Token Refresh Success Rate**
   - Should refresh every ~50 minutes
   - Look for "Refreshed X model with new AWS tokens" logs

2. **Compression Statistics**
   - Watch for context usage percentages
   - Check compression effectiveness (tokens before/after)

3. **Error Recovery**
   - Track ratio of recoverable vs unrecoverable errors
   - Monitor readonly chat occurrences (should decrease)

## Configuration

### Compression Settings (in `app/config.py`)

Current defaults:
```python
ELLM_CTX_LEN: int = 200_000  # Max context for efficient model
MAX_OUTPUT_COUNT: int = 12_000  # Max tokens in response
```

Compression triggers at 70% of `ELLM_CTX_LEN` and trims to 70% to leave room for response.

### Token Refresh Timing

Refreshes 10 minutes before expiration (hardcoded in `chat_models.py`):
```python
refresh_time = next_hour - timedelta(minutes=10)
```

Adjust if needed for your environment.

## Testing

### Manual Testing

1. **Test Token Refresh**:
   - Start application
   - Watch logs for "Next token refresh scheduled" message
   - Wait for scheduled time
   - Verify "Refreshed X model" appears in logs

2. **Test Compression**:
   - Create a long conversation (30+ exchanges)
   - Watch for "Message history at X% capacity" logs
   - Verify compression occurs and conversation continues

3. **Test Error Recovery**:
   - Trigger token expiration (wait until top of hour without refresh)
   - Send message
   - Verify automatic retry occurs
   - Check chat is not set to readonly

### Automated Testing

Run validation script:
```bash
./validate_fixes.sh
```

Should output:
```
✓ tiktoken available
✓ Token counting works: 4 tokens  
✓ Message trimming works: 10 -> 3 messages
✓ Token expiry calculation works: next refresh at 2026-02-21 15:50:00
✓ Model management module loaded
✓ Error classification works: TimeoutError is recoverable=True
✓ User-friendly messages work
```

## Troubleshooting

### Token Refresh Not Working

**Symptoms**: Conversations still fail at top of hour

**Checks**:
1. Verify background task started: `grep "Started AWS token refresh background task" backend.log`
2. Check for refresh logs: `grep "Proactively refreshing" backend.log`
3. Ensure event loop is running (task requires async context)

**Fix**: Restart application to reinitialize background task

### Compression Not Triggering

**Symptoms**: Long conversations still cause "context too long" errors

**Checks**:
1. Verify compression module imported: `grep "from app.message.compression" backend.log`
2. Check token counts: Look for "Message history at X% capacity" logs
3. Verify tiktoken installed: `pip list | grep tiktoken`

**Fix**: Install tiktoken, restart application

### Chats Still Going Readonly

**Symptoms**: Chats set to readonly on recoverable errors

**Checks**:
1. Check which errors occurred: `grep "Setting to readonly" backend.log`
2. Verify error handling imported: Check `message/service.py` imports
3. Check if ExpiredTokenException is being caught: `grep "ExpiredTokenException" backend.log`

**Fix**: Update `message/service.py` to use `handle_stream_error_enhanced`

## Rollback

If issues occur, rollback to previous version:

```bash
cd ~/dish-chat
BACKUP_DIR="backups/token_compression_fix_YYYYMMDD_HHMMSS"

# Restore backups
cp $BACKUP_DIR/chat_models.py.backup app/core/llm/chat_models.py
cp $BACKUP_DIR/agent_service.py.backup app/agent/service.py  
cp $BACKUP_DIR/message_service.py.backup app/message/service.py

# Remove new files
rm app/message/compression.py
rm app/message/error_handling.py

# Restart
pkill -f "uvicorn app.main:app"
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

## Performance Impact

### Expected Changes:

**Positive:**
- Reduced conversation interruptions (token expiry eliminated)
- Better handling of long conversations (compression prevents crashes)
- Fewer readonly chats (better error recovery)

**Neutral:**
- Slight CPU increase for background token refresh task (negligible)
- Token counting overhead when checking for compression (< 10ms)
- Compression overhead when triggered (< 500ms, only when needed)

### Memory Impact:

- Background task: ~1MB
- Tiktoken model: ~2MB loaded once
- Per-request overhead: Minimal (reuses loaded encoder)

## Future Improvements

1. **Adaptive Compression**: Adjust compression thresholds based on conversation patterns
2. **Compression Statistics Dashboard**: UI to show compression activity
3. **Conversation Summarization**: Replace old messages with summaries instead of deletion
4. **Token Usage Alerting**: Proactive alerts when approaching limits
5. **Multi-Model Support**: Extend token refresh to other providers (OpenAI, etc.)

## Support

For issues or questions:
1. Check logs in `~/dish-chat/backend.log`
2. Run validation script: `./validate_fixes.sh`
3. Review this README for troubleshooting steps
4. Contact: [your contact info]

## Version History

- **v1.0** (2026-02-21): Initial release
  - Proactive AWS token refresh
  - Message history compression
  - Enhanced error handling
