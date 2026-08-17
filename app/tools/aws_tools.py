# app/tools/aws_tools.py
"""
AWS tools for jakebot IAM permissions.
Read-only focused tools for Bedrock, S3, Athena, and Glue.
"""
import logging
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from langchain.tools import tool

from app.agent_mode.thought_interceptor import interceptor

logger = logging.getLogger(__name__)

# Initialize AWS clients (will use IAM role credentials automatically)
def _get_bedrock_client():
    """Get Bedrock Runtime client (for list_models only; invoke uses ChatBedrockConverse with profile ARNs)."""
    return boto3.client('bedrock-runtime', region_name='us-west-2')

def _get_s3_client():
    """Get S3 client."""
    return boto3.client('s3', region_name='us-west-2')

def _get_athena_client():
    """Get Athena client."""
    return boto3.client('athena', region_name='us-west-2')

def _get_glue_client():
    """Get Glue client."""
    return boto3.client('glue', region_name='us-west-2')


# ============================================================================
# BEDROCK TOOLS
# ============================================================================

@tool("bedrock_list_models")
def bedrock_list_models() -> str:
    """
    List available Bedrock foundation models.
    
    Returns a formatted list of model IDs and their capabilities.
    This helps users understand what models are available for inference.
    """
    interceptor.tool_call("bedrock_list_models", params={})
    interceptor.thought("Listing available Bedrock models", "tool")
    
    try:
        # Note: This uses bedrock client (not bedrock-runtime) for listing models
        bedrock_client = boto3.client('bedrock', region_name='us-west-2')
        response = bedrock_client.list_foundation_models()
        
        models = response.get('modelSummaries', [])
        
        if not models:
            return "No Bedrock models found or insufficient permissions."
        
        output = "Available Bedrock Models:\n" + "="*60 + "\n\n"
        
        for model in models[:20]:  # Limit to first 20
            model_id = model.get('modelId', 'Unknown')
            model_name = model.get('modelName', 'Unknown')
            provider = model.get('providerName', 'Unknown')
            
            output += f"• {model_name} ({provider})\n"
            output += f"  ID: {model_id}\n"
            output += f"  Modalities: {', '.join(model.get('inputModalities', []))} → {', '.join(model.get('outputModalities', []))}\n\n"
        
        if len(models) > 20:
            output += f"\n... and {len(models) - 20} more models\n"
        
        interceptor.tool_call("bedrock_list_models", result=f"Listed {len(models)} models")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error listing Bedrock models: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("bedrock_list_models", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing Bedrock models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("bedrock_list_models", result=f"Error: {error_msg}")
        return error_msg


@tool("bedrock_invoke_model")
def bedrock_invoke_model(model_id: str, prompt: str, max_tokens: int = 1000) -> str:
    """
    Invoke an Anthropic Claude model via Bedrock Application Inference Profile.
    
    Args:
        model_id: One of 'sonnet', 'haiku', 'opus', or a full profile ARN.
        prompt: The input prompt to send to the model.
        max_tokens: Maximum tokens to generate (default: 1000).
    
    Returns:
        The model's response text.
    
    Note: Direct Bedrock model IDs are no longer accepted.
          All calls route through Application Inference Profile ARNs.
    """
    interceptor.tool_call("bedrock_invoke_model", params={
        "model_id": model_id,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "max_tokens": max_tokens
    })
    interceptor.thought(f"Invoking Bedrock model via profile ARN: {model_id}", "tool")
    
    # Map friendly names to Application Inference Profile ARNs
    _PROFILE_MAP = {
        "sonnet": "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/5c511xksna83",
        "haiku":  "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/wpnvchycfust",
        "opus":   "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/m4hvzo6r2exy",
        "embed":  "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/4xgakngy389z",
    }
    
    # Resolve model_id to profile ARN
    if model_id.startswith("arn:aws:bedrock:"):
        profile_arn = model_id
    elif model_id in _PROFILE_MAP:
        profile_arn = _PROFILE_MAP[model_id]
    else:
        return (
            f"Error: '{model_id}' is not a valid profile alias or ARN. "
            f"Use one of: {list(_PROFILE_MAP.keys())} or a full application-inference-profile ARN."
        )
    
    from app.config import get_settings as _get_settings
    _settings = _get_settings()
    if _settings.PLLM_PROVIDER != "aws-bedrock":
        msg = (
            f"bedrock_invoke_model is only available when PLLM_PROVIDER='aws-bedrock'. "
            f"Current provider: {_settings.PLLM_PROVIDER!r}. Use the standard chat model instead."
        )
        interceptor.tool_call("bedrock_invoke_model", result=msg)
        return msg

    try:
        from langchain_core.messages import HumanMessage
        from app.core.llm.chat_models import _create_bedrock_model

        # Use the managed model factory so that Application Inference Profile
        # ARNs are enforced and credentials are always fresh.
        model = _create_bedrock_model(profile_arn, max_tokens)
        response = model.invoke([HumanMessage(content=prompt)])
        result_text = response.content if isinstance(response.content, str) else str(response.content)
        interceptor.tool_call("bedrock_invoke_model", result="Model invoked successfully via profile ARN")
        return "Model Response:\n" + result_text

    except Exception as e:
        error_msg = f"Error invoking Bedrock model: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("bedrock_invoke_model", result=f"Error: {error_msg}")
        return error_msg


# ============================================================================
# S3 TOOLS (Read-focused)
# ============================================================================

@tool("s3_list_buckets")
def s3_list_buckets() -> str:
    """
    List all S3 buckets accessible to this account.
    
    Returns a formatted list of bucket names and creation dates.
    """
    interceptor.tool_call("s3_list_buckets", params={})
    interceptor.thought("Listing S3 buckets", "tool")
    
    try:
        client = _get_s3_client()
        response = client.list_buckets()
        
        buckets = response.get('Buckets', [])
        
        if not buckets:
            return "No S3 buckets found or insufficient permissions."
        
        output = f"S3 Buckets ({len(buckets)} total):\n" + "="*60 + "\n\n"
        
        for bucket in buckets:
            name = bucket['Name']
            created = bucket['CreationDate'].strftime('%Y-%m-%d %H:%M:%S')
            output += f"• {name}\n  Created: {created}\n\n"
        
        interceptor.tool_call("s3_list_buckets", result=f"Listed {len(buckets)} buckets")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error listing S3 buckets: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("s3_list_buckets", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing S3 buckets: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("s3_list_buckets", result=f"Error: {error_msg}")
        return error_msg


@tool("s3_list_objects")
def s3_list_objects(bucket: str, prefix: str = "", max_keys: int = 100) -> str:
    """
    List objects in an S3 bucket with optional prefix filter.
    
    Args:
        bucket: The S3 bucket name
        prefix: Optional prefix to filter objects (like a folder path)
        max_keys: Maximum number of objects to return (default: 100)
    
    Returns:
        A formatted list of object keys, sizes, and last modified dates
    
    Example:
        s3_list_objects("my-bucket", "data/2024/", 50)
    """
    interceptor.tool_call("s3_list_objects", params={
        "bucket": bucket,
        "prefix": prefix,
        "max_keys": max_keys
    })
    interceptor.thought(f"Listing objects in s3://{bucket}/{prefix}", "tool")
    
    try:
        client = _get_s3_client()
        
        params = {
            'Bucket': bucket,
            'MaxKeys': max_keys
        }
        if prefix:
            params['Prefix'] = prefix
        
        response = client.list_objects_v2(**params)
        
        objects = response.get('Contents', [])
        
        if not objects:
            return f"No objects found in s3://{bucket}/{prefix}"
        
        output = f"Objects in s3://{bucket}/{prefix}\n"
        output += f"Total: {response.get('KeyCount', 0)} objects\n"
        output += "="*80 + "\n\n"
        
        for obj in objects:
            key = obj['Key']
            size = obj['Size']
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Format size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024**2:
                size_str = f"{size/1024:.1f} KB"
            elif size < 1024**3:
                size_str = f"{size/1024**2:.1f} MB"
            else:
                size_str = f"{size/1024**3:.1f} GB"
            
            output += f"• {key}\n"
            output += f"  Size: {size_str}  |  Modified: {modified}\n\n"
        
        if response.get('IsTruncated', False):
            output += "\n(Results truncated - more objects available)\n"
        
        interceptor.tool_call("s3_list_objects", result=f"Listed {len(objects)} objects")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error listing S3 objects: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("s3_list_objects", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing S3 objects: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("s3_list_objects", result=f"Error: {error_msg}")
        return error_msg


@tool("s3_get_object")
def s3_get_object(bucket: str, key: str, max_bytes: int = 10000) -> str:
    """
    Read the contents of an S3 object (text files only).
    
    Args:
        bucket: The S3 bucket name
        key: The object key (file path within the bucket)
        max_bytes: Maximum bytes to read (default: 10000 to prevent large files)
    
    Returns:
        The file contents as text (truncated if larger than max_bytes)
    
    Example:
        s3_get_object("my-bucket", "config/settings.json")
    
    Note: Only suitable for text files. Binary files will show garbled output.
    """
    interceptor.tool_call("s3_get_object", params={
        "bucket": bucket,
        "key": key,
        "max_bytes": max_bytes
    })
    interceptor.thought(f"Reading s3://{bucket}/{key}", "tool")
    
    try:
        client = _get_s3_client()
        
        # Get object metadata first
        head = client.head_object(Bucket=bucket, Key=key)
        file_size = head['ContentLength']
        
        # Read object
        response = client.get_object(Bucket=bucket, Key=key)
        
        # Read up to max_bytes
        content = response['Body'].read(max_bytes)
        
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            return f"Error: s3://{bucket}/{key} appears to be a binary file, not text. Use S3 console for binary files."
        
        output = f"Contents of s3://{bucket}/{key}\n"
        output += f"File size: {file_size} bytes\n"
        output += "="*80 + "\n\n"
        output += content_str
        
        if file_size > max_bytes:
            output += f"\n\n... (truncated, showing first {max_bytes} bytes of {file_size} total)"
        
        interceptor.tool_call("s3_get_object", result="Object read successfully")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error reading S3 object: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("s3_get_object", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error reading S3 object: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("s3_get_object", result=f"Error: {error_msg}")
        return error_msg


# ============================================================================
# ATHENA TOOLS
# ============================================================================

@tool("athena_list_databases")
def athena_list_databases(catalog: str = "AwsDataCatalog") -> str:
    """
    List databases in AWS Glue Data Catalog for Athena queries.
    
    Args:
        catalog: The catalog name (default: "AwsDataCatalog")
    
    Returns:
        A formatted list of database names
    """
    interceptor.tool_call("athena_list_databases", params={"catalog": catalog})
    interceptor.thought(f"Listing Athena databases in catalog: {catalog}", "tool")
    
    try:
        glue_client = _get_glue_client()
        response = glue_client.get_databases(CatalogId=catalog)
        
        databases = response.get('DatabaseList', [])
        
        if not databases:
            return f"No databases found in catalog: {catalog}"
        
        output = f"Athena Databases in {catalog}:\n" + "="*60 + "\n\n"
        
        for db in databases:
            name = db['Name']
            description = db.get('Description', 'No description')
            output += f"• {name}\n  {description}\n\n"
        
        interceptor.tool_call("athena_list_databases", result=f"Listed {len(databases)} databases")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error listing databases: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("athena_list_databases", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing databases: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("athena_list_databases", result=f"Error: {error_msg}")
        return error_msg


@tool("athena_list_tables")
def athena_list_tables(database: str) -> str:
    """
    List tables in an Athena database.
    
    Args:
        database: The database name
    
    Returns:
        A formatted list of table names and their descriptions
    """
    interceptor.tool_call("athena_list_tables", params={"database": database})
    interceptor.thought(f"Listing tables in database: {database}", "tool")
    
    try:
        glue_client = _get_glue_client()
        response = glue_client.get_tables(DatabaseName=database)
        
        tables = response.get('TableList', [])
        
        if not tables:
            return f"No tables found in database: {database}"
        
        output = f"Tables in {database}:\n" + "="*60 + "\n\n"
        
        for table in tables:
            name = table['Name']
            columns = table.get('StorageDescriptor', {}).get('Columns', [])
            col_count = len(columns)
            
            output += f"• {name} ({col_count} columns)\n"
            
            # Show first few columns
            if columns:
                output += "  Columns: "
                col_names = [f"{c['Name']} ({c['Type']})" for c in columns[:5]]
                output += ", ".join(col_names)
                if col_count > 5:
                    output += f", ... and {col_count - 5} more"
                output += "\n"
            
            output += "\n"
        
        interceptor.tool_call("athena_list_tables", result=f"Listed {len(tables)} tables")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error listing tables: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("athena_list_tables", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("athena_list_tables", result=f"Error: {error_msg}")
        return error_msg


@tool("athena_execute_query")
def athena_execute_query(
    query: str,
    database: str,
    output_location: str = "s3://aws-athena-query-results-dish-chat/",
    max_wait_seconds: int = 60
) -> str:
    """
    Execute an Athena SQL query and return results.
    
    Args:
        query: The SQL query to execute (SELECT queries only for safety)
        database: The database to query against
        output_location: S3 location for query results (optional)
        max_wait_seconds: Maximum time to wait for query completion (default: 60)
    
    Returns:
        Query results formatted as a table
    
    Example:
        athena_execute_query("SELECT * FROM users LIMIT 10", "my_database")
    
    Security: Only SELECT queries are allowed. Queries with INSERT, UPDATE, 
    DELETE, DROP, CREATE, ALTER will be rejected.
    """
    interceptor.tool_call("athena_execute_query", params={
        "query": query[:100] + "..." if len(query) > 100 else query,
        "database": database
    })
    interceptor.thought(f"Executing Athena query in {database}", "tool")
    
    # Safety check: Only allow SELECT queries
    query_upper = query.strip().upper()
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    
    if not query_upper.startswith('SELECT'):
        return "Error: Only SELECT queries are allowed for safety. This tool is read-only."
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"Error: Query contains dangerous keyword '{keyword}'. Only SELECT queries are allowed."
    
    try:
        client = _get_athena_client()
        
        # Start query execution
        response = client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': output_location}
        )
        
        query_execution_id = response['QueryExecutionId']
        
        # Wait for query to complete
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            status_response = client.get_query_execution(QueryExecutionId=query_execution_id)
            status = status_response['QueryExecution']['Status']['State']
            
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            
            time.sleep(2)
        
        if status != 'SUCCEEDED':
            reason = status_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            return f"Query {status}: {reason}"
        
        # Get query results
        results_response = client.get_query_results(QueryExecutionId=query_execution_id, MaxResults=100)
        
        rows = results_response['ResultSet']['Rows']
        
        if not rows:
            return "Query returned no results"
        
        # First row is headers
        headers = [col.get('VarCharValue', '') for col in rows[0]['Data']]
        
        # Format as table
        output = f"Athena Query Results\n"
        output += f"Database: {database}\n"
        output += f"Query: {query[:100]}...\n" if len(query) > 100 else f"Query: {query}\n"
        output += "="*80 + "\n\n"
        
        # Header row
        output += " | ".join(headers) + "\n"
        output += "-" * 80 + "\n"
        
        # Data rows
        for row in rows[1:]:  # Skip header row
            values = [col.get('VarCharValue', '') for col in row['Data']]
            output += " | ".join(values) + "\n"
        
        result_count = len(rows) - 1  # Subtract header
        if results_response.get('NextToken'):
            output += f"\n(Showing first {result_count} rows - more results available)\n"
        else:
            output += f"\n({result_count} rows total)\n"
        
        interceptor.tool_call("athena_execute_query", result=f"Query succeeded, {result_count} rows")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error executing Athena query: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("athena_execute_query", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error executing Athena query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("athena_execute_query", result=f"Error: {error_msg}")
        return error_msg


# ============================================================================
# GLUE CATALOG TOOLS
# ============================================================================

@tool("glue_get_table_schema")
def glue_get_table_schema(database: str, table: str) -> str:
    """
    Get the schema (columns and metadata) for a Glue catalog table.
    
    Args:
        database: The database name
        table: The table name
    
    Returns:
        Detailed schema information including columns, partitions, and storage details
    """
    interceptor.tool_call("glue_get_table_schema", params={
        "database": database,
        "table": table
    })
    interceptor.thought(f"Getting schema for {database}.{table}", "tool")
    
    try:
        glue_client = _get_glue_client()
        response = glue_client.get_table(DatabaseName=database, Name=table)
        
        table_data = response['Table']
        storage_desc = table_data.get('StorageDescriptor', {})
        
        output = f"Schema for {database}.{table}\n" + "="*80 + "\n\n"
        
        # Table properties
        output += "Table Properties:\n"
        output += f"  Location: {storage_desc.get('Location', 'N/A')}\n"
        output += f"  Input Format: {storage_desc.get('InputFormat', 'N/A')}\n"
        output += f"  Output Format: {storage_desc.get('OutputFormat', 'N/A')}\n"
        output += f"  Compressed: {storage_desc.get('Compressed', False)}\n"
        output += "\n"
        
        # Columns
        columns = storage_desc.get('Columns', [])
        output += f"Columns ({len(columns)} total):\n"
        output += "-" * 80 + "\n"
        
        for col in columns:
            col_name = col['Name']
            col_type = col['Type']
            col_comment = col.get('Comment', '')
            output += f"  • {col_name:<30} {col_type:<20}"
            if col_comment:
                output += f" -- {col_comment}"
            output += "\n"
        
        # Partition keys
        partition_keys = table_data.get('PartitionKeys', [])
        if partition_keys:
            output += "\n" + "Partition Keys:\n"
            for pk in partition_keys:
                output += f"  • {pk['Name']} ({pk['Type']})\n"
        
        # Parameters
        params = table_data.get('Parameters', {})
        if params:
            output += "\n" + "Table Parameters:\n"
            for key, value in list(params.items())[:10]:  # Show first 10
                output += f"  {key}: {value}\n"
        
        interceptor.tool_call("glue_get_table_schema", result="Schema retrieved successfully")
        return output
        
    except ClientError as e:
        error_msg = f"AWS Error getting table schema: {e.response['Error']['Message']}"
        logger.error(error_msg)
        interceptor.tool_call("glue_get_table_schema", result=f"Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error getting table schema: {str(e)}"
        logger.error(error_msg, exc_info=True)
        interceptor.tool_call("glue_get_table_schema", result=f"Error: {error_msg}")
        return error_msg
