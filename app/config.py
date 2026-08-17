from datetime import timedelta
import os
from app.tools.http_utils import make_noverify_http_client
from typing import Literal, Optional, ClassVar
from functools import lru_cache, cached_property
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ConfigDict
from pydantic import computed_field, Field

# AWS SigV4 auth for MCP tools behind IAM-protected Lambda Function URLs
from app.sigv4_auth import AWSSigV4Auth


def _json_headers(**extra: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    headers.update({k: v for k, v in extra.items() if v})
    return headers


def _bearer_headers(env_name: str, **extra: str) -> dict[str, str]:
    """Build headers from an environment-sourced bearer token."""
    headers = _json_headers(**extra)
    token = os.getenv(env_name, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@lru_cache
def get_settings():
    return Settings()


class Settings(BaseSettings):
    NAME: str = "Dish-Chat"
    VERSION: str = "2.1.0"
    AGENT_THOUGHT_CAPTURE_ENABLED: bool = Field(default=False)
    AGENT_VIZ_SERVER_URL: str = Field(default="http://localhost:8000/rest/api/v1/viz/event")
    API_PREFIX: str = "/rest/api/v1"

    # Runtime settings
    DEBUG: bool = False
    LOCAL: bool = True
    
    # Authentication Settings
    AUTH_DISABLED: bool = False  # Set to True to disable auth requirement (dev/testing only)
    DEFAULT_USER_EMAIL: str = "test.user@dish.com"  # Default email when auth is disabled
    ECHO_SQL: bool = False
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    CLEANUP_TIMEOUT: int = 600
    
    # Idle chat checker settings
    IDLE_CHAT_CHECKER_ENABLED: bool = True
    IDLE_CHAT_CHECK_INTERVAL_MINUTES: int = 1440  # Once per day (24h)
    IDLE_CHAT_THRESHOLD_MINUTES: int = 30  # Minutes of inactivity before triggering journal
    IDLE_CHAT_MIN_MESSAGES: int = 5  # Minimum messages required for journal generation
    SPOOLED_MAX_SIZE: int = 2 * 1024 * 1024
    # Must be supplied by environment in real deployments.
    MASTER_KEY: str = Field(default_factory=lambda: os.getenv("MASTER_KEY", ""))

    # DB
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "dishchat"
    POSTGRES_USER: str = "dev_user"
    POSTGRES_PWD: str = "dev123"

    @computed_field
    @cached_property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PWD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @computed_field
    @cached_property
    def POSTGRES_SQLALCHEMY_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PWD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # LLM Models - P for Power and E for Efficient
    PLLM_PROVIDER: Literal["aws-bedrock", "openai", "anthropic", "ollama"] = "ollama"
    PLLM_API_BASE: Optional[str] = "http://10.79.85.35:11434"
    PLLM_MODEL: str = "deepseek-r1:32b"
    # Tool-capable model for Ollama (used when tools are bound)
    PLLM_TOOL_MODEL: str = "llama3.2:latest"
    PLLM_CTX_LEN: int = 32_768
    ELLM_PROVIDER: Optional[Literal["aws-bedrock", "openai", "anthropic", "ollama"]] = "ollama"
    ELLM_API_BASE: Optional[str] = "http://10.79.85.35:11434"
    ELLM_MODEL: Optional[str] = "deepseek-r1:32b"
    ELLM_TOOL_MODEL: Optional[str] = "llama3.2:latest"
    ELLM_CTX_LEN: Optional[int] = 32_768
    LLM_TOKENIZER: Optional[str] = None  # HuggingFace Model ID or OpenAI
    DEFAULT_TEMP: float = 0.6
    DEFAULT_REASONING: bool = False
    REASONING_BUDGET: int = 6000

    # Provider-neutral model roles for Ollama-native orchestration.
    # MODEL_ROLES_CONFIG can be supplied as JSON and role env vars can override:
    # MODEL_ROLE_VERIFIER_MODEL, MODEL_ROLE_VERIFIER_PROVIDER,
    # MODEL_ROLE_VERIFIER_CONTEXT_LENGTH, MODEL_ROLE_VERIFIER_MAX_OUTPUT_TOKENS.
    OLLAMA_KEEP_ALIVE: str = "1h"
    OLLAMA_CTX_LEN: int = 65_536
    OLLAMA_MAX_OUTPUT_TOKENS: int = 8_192
    OLLAMA_TEMPERATURE: Optional[float] = None
    OLLAMA_REASONING: Optional[bool] = None
    MODEL_ROLES_CONFIG: dict = Field(default_factory=dict)

    # Optional aliases preserve existing deployment env names while active code
    # can request provider-neutral roles.
    MODEL_ROLE_PRIMARY_ALIAS: str = "primary"
    MODEL_ROLE_EFFICIENT_ALIAS: str = "efficient"
    MODEL_ROLE_COMPLEX_ALIAS: str = "complex"
    MODEL_ROLE_TOOL_WORKER_ALIAS: str = "tool_worker"
    MODEL_ROLE_ANALYST_ALIAS: str = "analyst"
    MODEL_ROLE_VERIFIER_ALIAS: str = "verifier"
    MODEL_ROLE_TITLE_ALIAS: str = "title"
    MODEL_ROLE_SUMMARY_ALIAS: str = "summary"

    # Embeddings
    EMBED_PROVIDER: Literal["aws-bedrock", "openai", "ollama"] = "ollama"
    EMBED_API_BASE: Optional[str] = "http://10.79.85.35:11434"
    EMBED_MODEL: str = "nomic-embed-text:latest"
    EMBED_TOKENIZER: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBED_CHUNK_SIZE: int = 300
    EMBED_OVERLAP: int = 0
    EMBED_BATCH_SIZE: int = 64  # Max for nomic-embed-text
    SUMMARY_LEN: int = 300

    # AWS Settings:
    AWS_REGION: Optional[str] = "us-west-2"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_FILESTORE_BUCKET: str = "dish-chat-attachments"
    
    # Local Storage Configuration (alternative to S3)
    USE_LOCAL_STORAGE_ONLY: str = "false"  # Set to "true" to use local storage instead of S3
    LOCAL_UPLOADS_DIR: str = "/tmp/dish-chat-uploads"  # Base directory for local file storage
    
    # AWS Bedrock Application Inference Profile
    BEDROCK_APPLICATION_INFERENCE_PROFILE_ARN: Optional[str] = None
    
    # Coverity Assist Integration (if using bearer token for external LLM calls)
    COVERITY_ASSIST_URL: Optional[str] = None
    COVERITY_ASSIST_TOKEN: Optional[str] = None
    
    # Sentry Integration (Added 2026-03-02)
    SENTRY_TOKEN: str = Field(default="", description="Sentry auth token")
    SENTRY_API_KEY: str = Field(default="", description="Sentry API key")
    
    # AWS Bedrock Configuration
    BEDROCK_READ_TIMEOUT: int = 300  # 5 minutes for long streaming responses
    BEDROCK_CONNECT_TIMEOUT: int = 10
    BEDROCK_MAX_RETRIES: int = 3

    # Chat Settings:
    MAX_CHAT_COUNT: int = 9999
    MAX_GROUPS_WITH_CHATS_COUNT: int = 20
    MAX_TITLE_LEN: int = 40
    MAX_TITLE_RETRY: int = 5
    # Max characters of stringified conversation to send to the title-gen model.
    # Keeps title generation well under the 200k Bedrock token limit even on
    # heavy investigation sessions (200k+ token conversations with large tool blobs).
    # 16000 chars ~ 4000 tokens — more than enough to understand the topic.
    TITLE_MAX_CONTEXT_CHARS: int = 16_000

    # Message Settings:
    MAX_VERSION_COUNT: int = 10
    MAX_IN_CTX_DOC_LEN: int = 4000
    MAX_OUTPUT_COUNT: int = 8_000  # Reduced for local model context window

    # Enhanced Attachment Settings with Log File Support
    SUPPORTED_DOC_TYPES: list[str] = [
        "application/pdf",
        "text/plain",
    # Markdown files
    "text/markdown",
    "text/x-markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # Log file MIME types
        "text/x-log",
        "application/x-log",
        "application/log",
        # Generic octet-stream for numbered/unknown extensions
        "application/octet-stream",
    ]
    
    # Log file extensions that should be treated as text/logs
    LOG_FILE_EXTENSIONS: list[str] = [
        ".log",
        ".txt",
        ".001", ".002", ".003", ".004", ".005",
        ".006", ".007", ".008", ".009", ".010",
        ".1", ".2", ".3", ".4", ".5",
        ".6", ".7", ".8", ".9", ".10",
        ".11", ".12", ".13", ".14", ".15",
        ".16", ".17", ".18", ".19", ".20",
        # Add more numbered extensions as needed
        ".trace",
        ".dump",
        ".out",
        ".err",
        ".debug",
        ".info",
        ".warn",
        ".error",
        ".cur",
        ".syslog",
        ".dmesg",
        ".messages",
    ]
    
    # Enable automatic log analysis
    AUTO_ANALYZE_LOGS: bool = True
    LOG_ANALYZER_TOOL_NAME: str = "logassist_embed_content"
    
    SUPPORTED_IMAGE_TYPES: list[str] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    ]
    MAX_IMAGE_RES: int = 1092 * 1092

    # Vault Settings:
    VAULT_SESSION_DURATION_SEC: int = 3 * 3600
    ARGON2_TIME: int = 3
    ARGON2_MEM: int = 2**16
    ARGON2_PRL: int = 1

    # Context Manager
    MAX_CONTEXT: int = 8_192  # Max length of core memory + conversation memory
    MAX_CONV_CACHE: float = 0.6  # Max proportion of conversation cache
    SUMMARIZE_WORD_LIMIT: int = 150
    CACHE_EVICT_PROP: float = 0.5  # Amount of conversation cache to evict each time

    # Agent Settings:
    MAX_CACHEPOINT_CNT: int = 4
    # LangGraph recursion limit (for agent tool loops)
    LANGGRAPH_RECURSION_LIMIT: int = 200  # Increased to handle complex agent workflows
    MAX_TOOL_CALLS_PER_TURN: int = 50  # Maximum consecutive tool calls before forcing response
    MAX_CONSECUTIVE_TOOL_ERRORS: int = 1555550  # Maximum tool errors before stopping


    # Beta report agent:
    BETAREPORT_MCP_CONFIG: dict = {
        "beta_report": {
            "transport": "streamable_http",
            "url": "https://7quifnvo576d2m5rhbnguwgvfq0qbivs.lambda-url.us-west-2.on.aws/mcp",
            # "url": "http://127.0.0.1:8001/mcp",
            "headers": _bearer_headers("BETAREPORT_MCP_BEARER_TOKEN"),
        }
    }


    # Individual MCP configurations (loaded separately)
    
    # JIRA MCP - Search JIRA issues
    JIRA_MCP_CONFIG: dict = {
        "jira_mcp": {
            "transport": "streamable_http",
            "url": "https://5qkld3ai7abi4f3kazn4ojkrzm0xqiji.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                **_bearer_headers("JIRA_MCP_BEARER_TOKEN"),
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        }
    }
    
    # Confluence MCP - Search Confluence documentation
    CONFLUENCE_MCP_CONFIG: dict = {
        "confluence_mcp": {
            "transport": "streamable_http",
            "url": "https://zkuyopzgcra6gsxceqclqbjtma0gfxlj.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                **_bearer_headers("CONFLUENCE_MCP_BEARER_TOKEN"),
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        }
    }
    
    # Netra MCP - Netra service integration
    # DISABLED: Conflicts with Net Detective MCP
    # NETRA_MCP_CONFIG: dict = {
        # "netra_mcp": {
            # "transport": "streamable_http",
            # "url": "https://qzjs7rodqibouzgyymup6jt25u0uqxqx.lambda-url.us-west-2.on.aws/mcp",
            # "headers": {
                # Authorization moved to NETRA_MCP_BEARER_TOKEN if this block is re-enabled.
                # "Content-Type": "application/json",
                # "Accept": "application/json, text/event-stream"
            # }
        # }
    # }


    # Net Detective MCP - ML-powered network diagnostics and error analysis
    # Viewership Measurement MCP:
    VIEWERSHIP_MCP_CONFIG: dict = {
        "viewership_measurement": {
            "transport": "streamable_http",
            "url": "https://cy4h556zxlhqyjju5psohdr6ou0scrxj.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        }
    }

    # Coverity Assist / log analysis MCP:
    LOG_ASSIST_MCP_CONFIG: dict = {
        "log_assist": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:5000/mcp",
        }
    }

    # Internal micro-tools MCP (SSO-protected):
    INTERNAL_TOOLS_MCP_CONFIG: dict = {
        "jira_mcp": {
            "transport": "streamable_http",
            "url": "https://5qkld3ai7abi4f3kazn4ojkrzm0xqiji.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                **_bearer_headers("JIRA_MCP_BEARER_TOKEN"),
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        },
        "confluence_mcp": {
            "transport": "streamable_http",
            "url": "https://zkuyopzgcra6gsxceqclqbjtma0gfxlj.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                **_bearer_headers("CONFLUENCE_MCP_BEARER_TOKEN"),
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        },
        # DISABLED DUE TO 502 ERROR - netra_mcp
        # "netra_mcp": {
        # "transport": "streamable_http",
        # "url": "https://qzjs7rodqibouzgyymup6jt25u0uqxqx.lambda-url.us-west-2.on.aws/mcp",
        # "headers": {
        # "Content-Type": "application/json",
        # "Accept": "application/json, text/event-stream"
        # }
        # }
    }
    # Net Detective MCP (ML-powered network analysis):
    NETDETECTIVE_MCP_CONFIG: dict = {
        "netdetective": {
            "transport": "streamable_http",
            "url": "http://localhost:8082/mcp",
            "timeout": 30,  # Analysis can take time
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        }
    }



    # Grasshopper MCP - STB log upload via Grasshopper SMP API
    # Auth: NONE — Lambda Function URL authorizationType=NONE (MR pending: fix-lambda-url-auth-response-stream-20260715)
    GRASSHOPPER_MCP_CONFIG: dict = {
        "grasshopper_mcp": {
            "transport": "streamable_http",
            "url": "https://a62nulaoz2dex55fzel2ovodji0cuzyu.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # S3 STB Logs MCP - Read STB diagnostic logs from S3 bucket
    # Auth: NONE — Lambda Function URL has authorizationType=NONE (public)
    S3_STB_LOGS_MCP_CONFIG: dict = {
        "s3_stb_logs": {
            "transport": "streamable_http",
            "url": "https://5hdyqxkr762nlbwxn2cdm56z740qbxbp.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # S3 STB Logs MCP (PROD) - Production variant with bounded safe limits
    # Auth: NONE — Lambda Function URL authorizationType=NONE (MR pending: fix-lambda-url-auth-response-stream-20260715)
    S3_STB_LOGS_PROD_MCP_CONFIG: dict = {
        "s3_stb_logs_prod": {
            "transport": "streamable_http",
            "url": "https://22uhsi3b5xmkwtemxht3sviuge0okjqt.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }



    # STB Health MCP - Direct bearer-auth Lambda (same URL as in k8s servers.yaml)
    STBHEALTH_MCP_CONFIG: dict = {
        "stbhealth_mcp": {
            "transport": "streamable_http",
            "url": "https://d4hlyjainms2apugt3uo236f540jujxl.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                **_bearer_headers("STBHEALTH_MCP_BEARER_TOKEN"),
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        }
    }

    # STB Health Popups MCP - Auth: AWS IAM SigV4
    STBHEALTH_POPUPS_MCP_CONFIG: dict = {
        "stbhealth_popups_mcp": {
            "transport": "streamable_http",
            "url": "https://p4hlb5scwsoatx2lbdyayoqcxy0qbtdn.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # RCA MCP Server - Root Cause Analysis pipelines, Auth: AWS IAM SigV4
    RCA_MCP_CONFIG: dict = {
        "rca_mcp": {
            "transport": "streamable_http",
            "url": "https://wq3pweejj4huv4rxzctjhhxkwm0llkvi.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # RTR Alerts MCP - RTR alert data and anomalies, Auth: AWS IAM SigV4
    RTR_ALERTS_MCP_CONFIG: dict = {
        "rtr_alerts_mcp": {
            "transport": "streamable_http",
            "url": "https://hfkme2yolbqhpbxbsf5eololy40badbf.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }


    # QoS MCP - QoS session analysis and OTA switchback diagnostics, Auth: AWS IAM SigV4
    QOS_MCP_CONFIG: dict = {
        "qos_mcp": {
            "transport": "streamable_http",
            "url": "https://37dmjuilmat2mozrkdgsiowi3u0orctw.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # dish-code-tools MCP - Structural code-intelligence tools for DISH STB firmware repos
    # 7 tools: browse_directory, read_file, search_regex, find_symbol,
    #          find_references, get_log, get_diff
    # Runs as a local process on the agent host (start-dish-code-tools.sh)
    # Repos: /tmp/home_agent/code_tools_dev/{stbctrl,DeviceManager,TvManager,src_tree}
    DISH_CODE_TOOLS_MCP_CONFIG: dict = {
        "dish_code_tools": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8087/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "timeout": timedelta(seconds=60),
        }
    }

    # DVA MCP - STB software jamming via dva-gateway on dsgpu3080 (10.79.85.47:5006)
    # Transport: plain HTTP to local network gateway (no AWS auth needed)
    # Tools: dva_list_stbs, dva_get_stb_info, dva_jam_software, dva_list_software
    DVA_MCP_CONFIG: dict = {
        "dva_mcp": {
            "transport": "streamable_http",
            "url": "http://10.79.85.47:5007/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "timeout": timedelta(seconds=750),
            "sse_read_timeout": timedelta(seconds=900),
        }
    }



    # Google Drive MCP - Read/search Google Drive files and documents
    # Auth: Service Account JSON file (GDRIVE_SERVICE_ACCOUNT_FILE) or OAuth2
    # Runs as a local process on the agent host (apps/gdrive_mcp/start.sh)
    # Tools: gdrive_search, gdrive_read_file, gdrive_list_folder,
    #        gdrive_get_file_metadata, gdrive_list_shared_drives, gdrive_auth_status
    GDRIVE_MCP_CONFIG: dict = {
        "gdrive_mcp": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8090/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        }
    }

    # ServiceNow ITSM MCP - Incident, change, and user search
    # Auth: SNOW_AUTH_MODE (oauth or basic) — credentials from environment
    # Runs as a local process on the agent host (apps/servicenow_mcp/start-mcp-server.sh)
    # Tools: snow_search_incidents, snow_get_incident, snow_search_changes,
    #        snow_get_change, snow_search_users, snow_query_table
    SERVICENOW_MCP_CONFIG: dict = {
        "servicenow_mcp": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8095/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        }
    }

    # EPG MCP - EPG schedule, STB-delivered EPG, channel/service metadata, Auth: AWS IAM SigV4
    EPG_MCP_CONFIG: dict = {
        "epg_mcp": {
            "transport": "streamable_http",
            "url": "https://4sozzchvwdkshyy5h2d4ovspc40gazwu.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # Net Detective MCP - ML analysis of Netra data for Dish STBs, Auth: AWS IAM SigV4
    NET_DETECTIVE_MCP_CONFIG: dict = {
        "net_detective_mcp": {
            "transport": "streamable_http",
            "url": "https://nldaym7ymkvhl3in2kdxe5dfuu0lloll.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=600),
        }
    }

    # Qodo Context Retriever MCP - Semantic code search across indexed DISH repos
    # Auth: Bearer JWT (from AWS Secrets Manager: qodo-mcp-dishchat)
    # Endpoint: qodo-ssh-proxy ClusterIP inside open-webui-dev namespace
    # Indexed repos: DT-ENG/DeviceManager, DT-DEVOPS/ansible, DT-ENG/src_tree,
    #                DT-ENG/stbctrl, DT-ENG/TvManager
    # Server: context_retriever v1.12.4 (Qodo AI)
    # Tools: get_context, deep_research, ask, list_repositories
    QODO_CONTEXT_MCP_CONFIG: dict = {
        "qodo_context_mcp": {
            "transport": "streamable_http",
            "url": "https://localhost:18443/mcp",
            "headers": {
                **_bearer_headers("QODO_CONTEXT_MCP_BEARER_TOKEN"),
                "Host": "qodo-context.dtc.dish.corp",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "timeout": timedelta(seconds=60),
            "sse_read_timeout": timedelta(seconds=120),
        }
    }
    # Headless Browser MCP - Web page rendering and interaction via Playwright/Chromium
    # Auth: NONE — Lambda Function URL authorizationType=NONE (MR pending: fix-lambda-url-auth-response-stream-20260715)
    HEADLESS_BROWSER_MCP_CONFIG: dict = {
        "headless_browser_mcp": {
            "transport": "streamable_http",
            "url": "https://zbjyvxkysl4ficlblm4uouwkd40synxy.lambda-url.us-west-2.on.aws/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "httpx_client_factory": make_noverify_http_client,
            "auth": AWSSigV4Auth(service="lambda", region="us-west-2"),
            "timeout": timedelta(seconds=120),
            "sse_read_timeout": timedelta(seconds=300),
        }
    }


    # External service endpoints
    COVERITY_GATEWAY_URL: str = "http://127.0.0.1:5001"

    # Sentry Integration (for cluster inspection and monitoring)
    SENTRY_AUTH_TOKEN: Optional[str] = None
    SENTRY_ORG: str = "dishtv.technology"
    SENTRY_URL: str = "https://ds-testing-sentry"
    SENTRY_PROJECT: Optional[str] = None  # Set if needed for specific project
    INTERNAL_SEARCH_URL: str = "https://internal-search.yourdomain/api/search"
    # Internal search mode and timeout
    INTERNAL_SEARCH_MODE: str = "multi"
    INTERNAL_SEARCH_TIMEOUT: float = 10.0
    INTERNAL_SEARCH_SOURCES: str = "confluence,gitlab,jira"

    # Confluence integration settings
    CONFLUENCE_BASE_URL: str = "https://dishtech-dishtv.atlassian.net/wiki"
    CONFLUENCE_USER_EMAIL: Optional[str] = None
    CONFLUENCE_API_TOKEN: Optional[str] = None

    # GitLab integration settings
    GITLAB_BASE_URL: str = "https://gitlab.com"
    GITLAB_TOKEN: Optional[str] = None
    GITLAB_SEARCH_SCOPES: str = "projects,blobs"

    # Jira integration settings
    JIRA_BASE_URL: str = "https://dishtech-dishtv.atlassian.net"
    JIRA_USER_EMAIL: Optional[str] = None
    JIRA_API_TOKEN: Optional[str] = None

    # Confluence integration settings
    # GitLab integration settings

    # Jira integration settings

    # Multi-source search settings




    # Provider-neutral model routing settings. Legacy OPUS_* names remain as
    # compatibility aliases for existing deployment configuration.
    COMPLEX_MODEL: str = "deepseek-r1:70b"
    OPUS_MODEL: str = COMPLEX_MODEL

    AUTO_MODEL_ROUTING_ENABLED: bool = True
    AUTO_OPUS_ENABLED: bool = AUTO_MODEL_ROUTING_ENABLED

    MODEL_ROUTING_COMPLEXITY_THRESHOLD: int = 8  # Score >= 8 required to route to Opus/complex model
    OPUS_COMPLEXITY_THRESHOLD: int = MODEL_ROUTING_COMPLEXITY_THRESHOLD

    MODEL_ROUTING_AB_TEST_ENABLED: bool = False
    OPUS_AB_TEST_ENABLED: bool = MODEL_ROUTING_AB_TEST_ENABLED

    MODEL_ROUTING_THRESHOLD_B: int = 4
    OPUS_THRESHOLD_B: int = MODEL_ROUTING_THRESHOLD_B

    # Optional overrides for specialized models
    AGENT_MODE_MODEL: Optional[str] = None
    AGENT_MODE_MAX_ITERS: int = 5
    # ──────────────────────────────────────────────────────────────────────
    # Multi-Conversation Orchestration Protocol (MCOP)
    # Allows the agent to spawn isolated child conversations for sub-tasks,
    # keeping the parent context lean while each child gets a fresh window.
    # ──────────────────────────────────────────────────────────────────────
    MCOP_ENABLED: bool = True
    MCOP_MAX_CHILDREN: int = 10          # max child tasks per parent run
    MCOP_CHILD_MAX_ITERS: int = 50       # iteration budget per child
    MCOP_PARALLEL_LIMIT: int = 10        # max concurrent child LLM calls
    MCOP_RESULT_MAX_TOKENS: int = 30000  # summary truncation (tokens)
    MCOP_CHILD_RESERVE_ITERS: int = 3   # iterations reserved for output at end of budget
    SUMMARY_MODEL_ARN: Optional[str] = None
    REVIEW_MODEL_ARN: Optional[str] = None

    # Read from .env - ALLOW EXTRA FIELDS from environment
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="allow",  # FIX: Allow extra fields from .env
    )
    __pydantic_config__ = ConfigDict(extra="allow")

    # Enable / disable MCP tool sets (handy for local dev)
    ENABLE_LOG_ASSIST_MCP: bool = False
    ENABLE_INTERNAL_TOOLS_MCP: bool = True   # Sentry cluster access enabled

    # Memory Enhancement
    ENABLE_MEMORY_INTEGRATION: bool = Field(default=False, description="Enable tool execution memory system")

    # CORS / frontend origins
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://10.79.83.40:3000",
        "http://10.79.83.40:3001",
        "http://10.79.85.47:3000",
        "http://10.79.85.47:3001",
        "http://10.79.85.47:8000",
        "http://10.79.85.47:8001",
        "http://10.79.85.35:3000",
        "http://10.79.85.35:3001",
        "http://0.0.0.0:3000",
        "http://0.0.0.0:3001",
        "http://0.0.0.0:3002",
        "http://10.79.85.35:3002",
        "https://chat-agent.dishtv.technology",
        "http://chat-agent.dishtv.technology"]
"""
Configuration Updates for Grasshopper Integration
==================================================

Add these settings to app/config.py in the Settings class:

"""

# Grasshopper STB Log Upload Configuration
GRASSHOPPER_HOST: str = Field(
    default="https://grasshopper-autoupload.dishanywhere.com:8443",
    description="Grasshopper SMP API host"
)
GRASSHOPPER_PORT: str = Field(
    default="8443",
    description="Grasshopper SMP API port"
)
GRASSHOPPER_AUTH_KEY: str = Field(
    default_factory=lambda: os.getenv("GRASSHOPPER_AUTH_KEY", ""),
    description="Grasshopper authentication key supplied by environment"
)

# OAuth Configuration (Production)
GRASSHOPPER_OAUTH_ENABLED: bool = Field(
    default=False,
    description="Enable OAuth 2.0 authentication for Grasshopper"
)
GRASSHOPPER_OAUTH_TOKEN_URL: Optional[str] = Field(
    default=None,
    description="OAuth token endpoint URL"
)
GRASSHOPPER_OAUTH_CLIENT_ID: Optional[str] = Field(
    default=None,
    description="OAuth client ID"
)
GRASSHOPPER_OAUTH_CLIENT_SECRET: Optional[str] = Field(
    default=None,
    description="OAuth client secret"
)

# Grasshopper Fallback Endpoints
GRASSHOPPER_S3_UPLOAD: str = Field(
    default="https://ds-ghuh.dishtv.technology/upload",
    description="S3 fallback upload endpoint"
)
GRASSHOPPER_CCSHARE_UPLOAD: str = Field(
    default="https://stbAnalyticsDU.echostarbeta.com/cgi-bin/ghuh",
    description="CCShare fallback upload endpoint"
)
