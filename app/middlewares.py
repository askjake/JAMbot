import logging
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool
from fastapi import Request, Header

logger = logging.getLogger(__name__)


class LocalIdInjectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        headers = dict(request.scope["headers"])
        headers[b"x-auth-request-email"] = b"test.test@dish.com"
        request.scope["headers"] = [(k, v) for k, v in headers.items()]

        return await call_next(request)


class LogRespMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_body = await request.body()
        
        logger.debug(f"Request path: {request.method} {request.url.path}")
        logger.debug(f"Request params: {request.query_params}")
        logger.debug(f"Request body: {request_body}")
        
        response = await call_next(request)
        
        # Check if it's a streaming response by headers
        is_streaming = (
            response.headers.get("transfer-encoding") == "chunked" or
            "text/event-stream" in response.headers.get("content-type", "")
        )
        
        if is_streaming:
            logger.debug(f"Streaming response, status: {response.status_code}")
        else:
            try:
                response_body = [chunk async for chunk in response.body_iterator]
                response.body_iterator = iterate_in_threadpool(iter(response_body))
                if response_body:
                    logger.debug(f"response_body={response_body[0].decode()}")
            except:
                logger.debug(f"Could not decode response, status: {response.status_code}")
        
        return response


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Bind one server-owned request ID for tool-execution audit correlation.

    The identifier is generated server-side (or accepted from a strictly
    validated ``X-Request-ID`` header) and stored in a ContextVar-held mutable
    correlation object.  Graph nodes update that object; they never receive the
    request ID as a model-editable tool argument.

    The ContextVar is deliberately not reset in a ``finally`` block: streaming
    (SSE) chat responses continue producing tool calls after ``dispatch``
    returns, and each request already runs in its own context copy, so there is
    no cross-request leakage.
    """

    _SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{8,120}$")

    async def dispatch(self, request: Request, call_next):
        from app.agent.tool_execution_audit import (
            AuditRequestState,
            bind_audit_request_state,
            new_request_id,
        )

        incoming = request.headers.get("x-request-id") or ""
        request_id = incoming if self._SAFE_REQUEST_ID.match(incoming) else new_request_id()
        bind_audit_request_state(AuditRequestState(request_id=request_id))
        request.state.audit_request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
