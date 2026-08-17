MODEL_PRICING = {
    "Claude-3.5-Sonnet": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "Claude-3.7-Sonnet": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "Claude-Sonnet-4": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input": 0.0008 / 1000,
        "cache_read": 0.00008 / 1000,
        "cache_create": 0.001 / 1000,
        "output": 0.004 / 1000,
    },
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input": 0.0008 / 1000,
        "cache_read": 0.00008 / 1000,
        "cache_create": 0.001 / 1000,
        "output": 0.004 / 1000,
    },
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    # Application Inference Profile ARNs (added for Xiuyan's profile migration)
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/5c511xksna83": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/wpnvchycfust": {
        "input": 0.003 / 1000,
        "cache_read": 0.0003 / 1000,
        "cache_create": 0.00375 / 1000,
        "output": 0.015 / 1000,
    },
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/4xgakngy389z": {
        "input": 0.0008 / 1000,
        "cache_read": 0.00008 / 1000,
        "cache_create": 0.001 / 1000,
        "output": 0.004 / 1000,
    },
    # Opus 4.1 Application Inference Profile (Auto-Opus routing)
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/m4hvzo6r2exy": {
        "input": 0.015 / 1000,        # $15 per MTok
        "cache_read": 0.0015 / 1000,  # $1.50 per MTok (10% of input)
        "cache_create": 0.01875 / 1000,  # $18.75 per MTok (1.25x input)
        "output": 0.075 / 1000,       # $75 per MTok
    },
    # Local/Ollama models (zero cost - self-hosted)
    "llama3.2:latest": {
        "input": 0.0,
        "cache_read": 0.0,
        "cache_create": 0.0,
        "output": 0.0,
    },
}

# DEPRECATED: Cross-region inference profiles (untagged - replaced by application inference profiles)
# Kept for backward-compatible pricing lookup of historical usage data only.
MODEL_PRICING["arn:aws:bedrock:us-west-2:233532778289:inference-profile/us.anthropic.claude-sonnet-4-6"] = {
    "input": 0.003 / 1000,
    "cache_read": 0.0003 / 1000,
    "cache_create": 0.00375 / 1000,
    "output": 0.015 / 1000,
}
MODEL_PRICING["arn:aws:bedrock:us-west-2:233532778289:inference-profile/us.anthropic.claude-opus-4-6-v1"] = {
    "input": 0.015 / 1000,
    "cache_read": 0.0015 / 1000,
    "cache_create": 0.01875 / 1000,
    "output": 0.075 / 1000,
}
