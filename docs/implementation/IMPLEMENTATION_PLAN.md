
# Dish-Chat Token and Prompt Management Fix Summary
=================================================

## Issues Identified:

### 1. AWS Token Expiration
- AWS tokens expire at the top of the hour
- Conversations get interrupted and go into readonly mode
- No proactive token refresh mechanism
- Token retry exists but not used consistently

### 2. Prompt Token Overflow
- No message history compression/trimming
- Long conversations exceed model context limits
- Causes crashes and readonly mode

### 3. Poor Error Recovery
- Chats go readonly on any error
- No graceful degradation
- Lost context when errors occur

## Solutions to Implement:

### 1. Proactive AWS Token Refresh
- Add background task to refresh tokens before expiration (50 min)
- Force model recreation with new credentials
- Retry failed requests with fresh tokens
- Add token expiry tracking

### 2. Message History Compression
- Implement langchain's trim_messages utility
- Keep recent messages + system prompt
- Compress older messages via summarization
- Monitor token usage per request

### 3. Improved Error Handling
- Don't set readonly on recoverable errors
- Retry with token refresh on AWS errors
- Remove failed message and continue
- Add user notification without breaking chat

## Files to Modify:

1. app/core/llm/chat_models.py - Token management
2. app/agent/service.py - Message compression
3. app/message/service.py - Error handling
4. app/message/utils.py - Token counting
5. app/config.py - Add compression settings
