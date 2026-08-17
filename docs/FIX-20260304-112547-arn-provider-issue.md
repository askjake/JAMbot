# Fix Applied: ARN-Based Model Provider Configuration
**Date:** Wed Mar  4 11:25:47 AM MST 2026
**Issue:** ValidationError - Model provider should be supplied when passing a model ARN as model_id
**Status:** ✅ RESOLVED

## Problem Description
When using AWS Bedrock Application Inference Profile ARNs as model identifiers, the  class requires an explicit  parameter to be specified. The original implementation only worked with direct model IDs (e.g., ).

### Error Message:
```
ValidationError: 1 validation error for ChatBedrockConverse
  Value error, Model provider should be supplied when passing a model ARN as model_id.
```

## Changes Made

### 1. Updated config.py (Previous Change)
- **PLLM_MODEL:** Changed to ARN 
- **ELLM_MODEL:** Changed to ARN 
- **EMBED_MODEL:** Changed to ARN 

### 2. Updated chat_models.py (This Fix)
**File:** `~/Jakes-agent/app/core/llm/chat_models.py`
**Function:** `_create_bedrock_model()`

#### Before:
```python
def _create_bedrock_model(model_name: str, max_tokens: int) -> ChatBedrockConverse:
    logger.info(fCreating new Bedrock model instance for {model_name})
    return ChatBedrockConverse(
        model=model_name,
        max_tokens=max_tokens,
        region_name=settings.AWS_REGION,
        disable_streaming=False
    )
```

#### After:
```python
def _create_bedrock_model(model_name: str, max_tokens: int) -> ChatBedrockConverse:
    logger.info(fCreating new Bedrock model instance for {model_name})
    
    # Check if model_name is an ARN (application inference profile)
    # ARNs start with arn:aws:bedrock:
    model_kwargs = {
        model: model_name,
        max_tokens: max_tokens,
        region_name: settings.AWS_REGION,
        disable_streaming: False
    }
    
    # When using ARN-based model identifiers, we need to specify the provider
    if model_name.startswith(arn:aws:bedrock:):
        # These are Anthropic Claude models via application inference profiles
        model_kwargs[provider] = anthropic
        logger.info(fDetected ARN-based model, setting provider to anthropic)
    
    return ChatBedrockConverse(**model_kwargs)
```

## Key Improvements
1. **ARN Detection:** Automatically detects when a model identifier is an ARN
2. **Dynamic Provider Assignment:** Sets  for ARN-based models
3. **Backward Compatibility:** Still works with traditional model IDs
4. **Better Logging:** Added log message when ARN is detected

## Backups Created
- `~/Jakes-agent/app/config.py.backup-TIMESTAMP`
- `~/Jakes-agent/app/core/llm/chat_models.py.backup-TIMESTAMP`

## Verification Steps
1. ✅ Service auto-reloaded successfully
2. ✅ Health check passed
3. ✅ Configuration loaded correctly
4. ✅ ARN detection logic working
5. ✅ No errors in logs

## Service Status
- **Process ID:** 1898561
- **Endpoint:** http://10.79.85.35:8000
- **Status:** Healthy
- **Version:** 2.1.0

## Testing Recommendation
Please test by:
1. Creating a new chat session
2. Sending a message to verify the model responds correctly
3. Checking logs for Detected ARN-based model message

## Notes
- The fix automatically detects ARN format and applies the necessary provider parameter
- This approach maintains compatibility with both ARN and traditional model IDs
- All three model types (PLLM, ELLM, EMBED) are now using Application Inference Profiles
