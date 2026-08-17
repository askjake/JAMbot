"""AWS SigV4 authentication for httpx (used by langchain-mcp-adapters).

Provides an httpx.Auth subclass that signs every outbound HTTP request with
AWS Signature Version 4, enabling langchain-mcp-adapters to call Lambda
Function URLs that require AWS_IAM authentication.

Usage in config.py:
    from app.sigv4_auth import AWSSigV4Auth

    GRASSHOPPER_MCP_CONFIG = {
        "grasshopper_mcp": {
            "transport": "streamable_http",
            "url": "https://....lambda-url.us-west-2.on.aws/mcp",
            "auth": AWSSigV4Auth(),
        }
    }

Credential resolution uses botocore's standard chain:
  1. Environment variables (AWS_ACCESS_KEY_ID, etc.)
  2. Shared credentials file (~/.aws/credentials)
  3. IAM instance profile (EC2)
  4. EKS Pod Identity / IRSA (Kubernetes)

NOTE: This module intentionally has ZERO imports from app.* to avoid
circular imports (config.py imports this, and app.aws.clients imports config).
"""

from urllib.parse import urlparse

import httpx
from botocore.auth import SigV4Auth as _BotoSigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session as BotocoreSession

# Headers httpx manages internally - excluded from SigV4 signing because httpx
# may modify them AFTER auth_flow returns (e.g. content-length recalculation,
# connection pooling, encoding negotiation). Including them in the signature
# causes "signature mismatch" errors.
_HTTPX_MANAGED_HEADERS = frozenset({
    "host",
    "user-agent",
    "accept-encoding",
    "connection",
    "content-length",
    "transfer-encoding",
})


class AWSSigV4Auth(httpx.Auth):
    """httpx.Auth implementation that signs requests with AWS SigV4.

    Compatible with langchain-mcp-adapters StreamableHttpConnection `auth` parameter
    and the MCP SDK deprecated `streamablehttp_client(auth=...)`.
    """

    def __init__(self, service: str = "lambda", region: str = "us-west-2"):
        """
        Args:
            service: AWS service name for signing (default: "lambda").
                     Use "execute-api" for API Gateway.
            region:  AWS region for signing (default: "us-west-2").
        """
        self.service = service
        self.region = region

    def auth_flow(self, request: httpx.Request):
        """Sign the request and yield it back (httpx auth protocol)."""
        url = str(request.url)
        body = request.content if request.content else b""

        # Only sign headers we explicitly control - not httpx auto-generated ones.
        # httpx may modify content-length, connection, user-agent, etc. after
        # auth_flow returns, which would invalidate the signature.
        headers_to_sign = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HTTPX_MANAGED_HEADERS
        }
        # Host is REQUIRED by SigV4 - derive from URL
        headers_to_sign["Host"] = urlparse(url).netloc

        aws_request = AWSRequest(
            method=request.method,
            url=url,
            headers=headers_to_sign,
            data=body,
        )
        # Create a fresh BotocoreSession on every signing request so that
        # updated credentials written to ~/.aws/credentials by secgateway are
        # always picked up without requiring a process restart.
        credentials = BotocoreSession().get_credentials()
        _BotoSigV4Auth(credentials, self.service, self.region).add_auth(aws_request)

        # Apply only the auth-specific headers back to the request
        for key in ("X-Amz-Date", "X-Amz-Security-Token", "Authorization"):
            if key in aws_request.headers:
                request.headers[key] = aws_request.headers[key]

        yield request

    def __repr__(self) -> str:
        return f"AWSSigV4Auth(service={self.service!r}, region={self.region!r})"
