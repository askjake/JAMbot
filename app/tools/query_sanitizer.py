"""
Query sanitization for public web searches.
Prevents proprietary information from being sent to external services.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Patterns that indicate potentially sensitive information
SENSITIVE_PATTERNS = [
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@(?:dish|echostar)\.com\b', 'EMAIL'),
    # Internal domains
    (r'\b(?:[\w-]+\.)?(?:dish|echostar)\.(?:com|net|org)\b', 'INTERNAL_DOMAIN'),
    # IP addresses (private ranges)
    (r'\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d+\.\d+\b', 'PRIVATE_IP'),
    # Common internal system indicators
    (r'\b(?:RAIL|DATASO|internal|intranet|vpn|prod|staging|dev)-[\w-]+\b', 'INTERNAL_SYSTEM'),
    # Employee IDs or similar
    (r'\b(?:EMP|USR|ID)[-_]?\d{4,}\b', 'EMPLOYEE_ID'),
    # API keys or tokens (basic patterns)
    (r'\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+\b', 'CREDENTIAL'),
]

# Company-specific terms that should trigger warnings
COMPANY_TERMS = [
    'dish', 'echostar', 'sling', 'boost',
    'rail', 'dataso', 'sse', 'dt',
]

def contains_sensitive_info(query: str) -> Tuple[bool, list[str]]:
    """
    Check if a query contains potentially sensitive information.
    
    Returns:
        Tuple of (is_sensitive, list_of_violations)
    """
    violations = []
    query_lower = query.lower()
    
    # Check for sensitive patterns
    for pattern, violation_type in SENSITIVE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            violations.append(violation_type)
    
    # Check for company-specific terms combined with technical terms
    technical_terms = ['cluster', 'server', 'database', 'api', 'endpoint', 'config', 'credentials']
    has_company = any(term in query_lower for term in COMPANY_TERMS)
    has_technical = any(term in query_lower for term in technical_terms)
    
    if has_company and has_technical:
        violations.append('COMPANY_TECHNICAL_COMBO')
    
    return len(violations) > 0, violations


def sanitize_query(query: str) -> Tuple[str, bool, list[str]]:
    """
    Sanitize a query for public web search.
    
    Returns:
        Tuple of (sanitized_query, was_modified, violations)
    """
    is_sensitive, violations = contains_sensitive_info(query)
    
    if not is_sensitive:
        return query, False, []
    
    # Log the attempt
    logger.warning(
        f"Potentially sensitive query detected: {query[:50]}... "
        f"Violations: {violations}"
    )
    
    # For now, we'll return the original query but flag it
    # In production, you might want to block it entirely or auto-sanitize
    return query, True, violations


def should_block_query(query: str) -> Tuple[bool, str]:
    """
    Determine if a query should be blocked entirely.
    
    Returns:
        Tuple of (should_block, reason)
    """
    is_sensitive, violations = contains_sensitive_info(query)
    
    # Block if critical violations detected
    critical_violations = {'EMAIL', 'CREDENTIAL', 'PRIVATE_IP'}
    if any(v in critical_violations for v in violations):
        reason = f"Query contains sensitive information: {', '.join(violations)}"
        return True, reason
    
    return False, ""
