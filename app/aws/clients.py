from functools import cache
import boto3.session
from app.config import get_settings

settings = get_settings()


def _make_session() -> boto3.session.Session:
    """Create a fresh boto3 session so credentials are re-read from disk."""
    return boto3.session.Session(region_name=settings.AWS_REGION)


def get_s3_client():
    """
    Return a fresh S3 client on every call.

    boto3 clients cache credentials internally, so creating a new client on
    each call is the simplest way to guarantee that the latest credentials
    (refreshed by secgateway) are used.  The overhead is negligible compared
    to the actual S3 network round-trip.
    """
    return _make_session().client('s3')
