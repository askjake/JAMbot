import boto3
import json
import uuid
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any
from collections import defaultdict
from Crypto.Cipher import AES
from base64 import b64decode

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s,%(msecs)03d %(levelname)s %(pathname)s:%(lineno)d:%(funcName)s() %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PAGE_TABLE = os.environ.get("PAGE_TABLE")
MESSAGE_TABLE = os.environ.get("MESSAGE_TABLE")
USAGE_TABLE = os.environ.get("USAGE_TABLE")
MASTER_KEY = os.environ.get("MASTER_KEY")

required_env = [PAGE_TABLE, MESSAGE_TABLE, USAGE_TABLE, MASTER_KEY]
if not all(required_env):
    missing = [
        var
        for var, val in zip(
            ["PAGE_TABLE", "MESSAGE_TABLE", "USAGE_TABLE", "MASTER_KEY"], required_env
        )
        if not val
    ]
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}"
    )


def decipher(message: str, b64key: str) -> str:
    """
    Decrypt message with AES-256 in GCM mode.
    Return plaintext
    """

    key = b64decode(b64key)
    if len(key) != 32:
        raise ValueError("Encryption key must be 256 bits.")

    b64tag, b64nonce, b64ciphertext = message.split("$")[2:]
    tag = b64decode(b64tag)
    nonce = b64decode(b64nonce)
    ciphertext = b64decode(b64ciphertext)

    cipher = AES.new(key, AES.MODE_GCM, nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)

    return data.decode("utf-8")


def extract_chat_data(users: List[str] = []) -> Dict[str, Dict[str, Any]]:
    """
    Extract chat data from DynamoDB tables and convert to intermediate format.

    Args:
        users: List of user IDs to extract. If empty, extracts all users.

    Returns:
        Dict with structure: {user_id: {chat_id: chat_object}
        Example:
        {
            "user.id.123": {
                "chat-uuid-456": {
                "user_id": "user.id.123",
                "chat_id": "chat-uuid-456",
                "title": "Chat Title",
                "creation_time": datetime(UTC),
                "last_message_at": datetime(UTC),
                "messages": {
                    "msg-uuid-789": {
                    "message_id": "msg-uuid-789",
                    "chat_id": "chat-uuid-456",
                    "role": "user|assistant",
                    "content": "decrypted content",
                    "timestamp": datetime(UTC),
                    "thinking": "decrypted thinking" | None,
                    "params": {parsed_dict} | None
                    },
                    ...
                },
                "usages": [
                    {
                    "user_id": "user.id.123",
                    "chat_id": "chat-uuid-456",
                    "timestamp": datetime(UTC),
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "model": "claude-3-sonnet"
                    },
                    ...
                ]
                }
            }
        }

    """
    # Get master key from environment
    MASTER_KEY = os.environ.get("MASTER_KEY")
    if not MASTER_KEY:
        raise ValueError("MASTER_KEY environment variable not set")

    # Initialize DynamoDB connection
    dynamodb = boto3.resource("dynamodb", "us-west-2")
    page_table = dynamodb.Table(PAGE_TABLE)
    msg_table = dynamodb.Table(MESSAGE_TABLE)
    usage_table = dynamodb.Table(USAGE_TABLE)

    result = {}

    # Step 1: Get usage data for all target users
    all_usage_data = defaultdict(list)

    if not users:
        # Scan entire usage table if no specific users provided
        logger.info("Scanning all usage data...")
        scan_kwargs = {}
        while True:
            response = usage_table.scan(**scan_kwargs)
            for item in response["Items"]:
                user_id = item.get("UserId")
                if user_id:  # Ensure UserId exists
                    all_usage_data[user_id].append(item)

            if "LastEvaluatedKey" not in response:
                break
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        users_to_process = list(all_usage_data.keys())
    else:
        # Query usage for specific users only
        users_to_process = users
        for user_id in users:
            logger.info(f"Querying usage data for user: {user_id}")
            try:
                query_kwargs = {
                    "KeyConditionExpression": "UserId = :user_id",
                    "ExpressionAttributeValues": {":user_id": user_id},
                }

                # Handle pagination for usage query
                while True:
                    response = usage_table.query(**query_kwargs)
                    all_usage_data[user_id].extend(response["Items"])

                    if "LastEvaluatedKey" not in response:
                        break
                    query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            except Exception as e:
                logger.error(f"Error querying usage for user {user_id}: {e}")
                continue

    # Step 2: Get page data
    all_page_data = defaultdict(list)

    if not users:
        # Scan entire page table if processing all users
        logger.info("Scanning all page data...")
        scan_kwargs = {}
        while True:
            response = page_table.scan(**scan_kwargs)
            for item in response["Items"]:
                user_id = item.get("UserId")
                if user_id:  # Ensure UserId exists
                    all_page_data[user_id].append(item)

            if "LastEvaluatedKey" not in response:
                break
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    else:
        # Query pages for specific users
        for user_id in users_to_process:
            logger.info(f"Querying page data for user: {user_id}")
            try:
                query_kwargs = {
                    "KeyConditionExpression": "UserId = :user_id",
                    "ExpressionAttributeValues": {":user_id": user_id},
                }

                # Handle pagination for page query
                while True:
                    response = page_table.query(**query_kwargs)
                    all_page_data[user_id].extend(response["Items"])

                    if "LastEvaluatedKey" not in response:
                        break
                    query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            except Exception as e:
                logger.error(f"Error querying pages for user {user_id}: {e}")
                continue

    # Step 3: Process each user
    for user_id in users_to_process:
        logger.info(f"Processing chats for user: {user_id}")
        user_chats = {}

        # Pre-index usage data by timestamp for efficient lookup
        usage_by_timestamp = {}
        for usage_item in all_usage_data[user_id]:
            try:
                timestamp_ms = int(usage_item["MessageTimestamp"])
                usage_by_timestamp[timestamp_ms] = {
                    "user_id": user_id,
                    "timestamp": datetime.fromtimestamp(
                        timestamp_ms / 1000, tz=timezone.utc
                    ),
                    "input_tokens": int(usage_item.get("inTkCnt", 0)),
                    "output_tokens": int(usage_item.get("outTkCnt", 0)),
                    "model": usage_item.get("model", "unknown"),
                }
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid usage record for user {user_id}: {e}")
                continue

        page_items = all_page_data.get(user_id, [])
        if not page_items:
            continue

        # Process each page for this user
        for page_item in page_items:
            chat_ids_data = page_item.get("ChatIds", [])

            for chat_info in chat_ids_data:
                # Safely parse chat metadata
                if not isinstance(chat_info, list) or len(chat_info) < 3:
                    logger.warning(f"Invalid chat_info format: {chat_info}")
                    continue

                try:
                    chat_id = chat_info[0]
                    title = chat_info[1]
                    creation_time = datetime.fromtimestamp(
                        int(chat_info[2]) / 1000, tz=timezone.utc
                    )
                    encryption_type = chat_info[3] if len(chat_info) > 3 else None
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing chat metadata: {e}")
                    continue

                # Skip vault encrypted chats
                if encryption_type == "vault":
                    logger.debug(f"Skipping vault chat: {chat_id}")
                    continue

                # Get messages for this chat with pagination
                try:
                    query_kwargs = {
                        "KeyConditionExpression": "ChatId = :chat_id",
                        "ExpressionAttributeValues": {":chat_id": chat_id},
                    }

                    messages_dict = {}
                    last_message_time = creation_time

                    # Handle pagination for message query
                    while True:
                        response = msg_table.query(**query_kwargs)

                        for msg_item in response["Items"]:
                            # Validate required fields
                            if "ChatMessage" not in msg_item:
                                logger.warning(
                                    f"Message missing ChatMessage field: {msg_item}"
                                )
                                continue

                            msg_data = msg_item["ChatMessage"]
                            required_fields = ["timestamp", "role", "content"]
                            if not all(field in msg_data for field in required_fields):
                                logger.warning(
                                    f"Message missing required fields: {msg_data}"
                                )
                                continue

                            try:
                                timestamp_ms = int(msg_data["timestamp"])
                                timestamp = datetime.fromtimestamp(
                                    timestamp_ms / 1000, tz=timezone.utc
                                )
                            except (ValueError, OSError) as e:
                                logger.warning(f"Invalid timestamp in message: {e}")
                                continue

                            # Decrypt content if needed
                            content = msg_data["content"]
                            thinking = msg_data.get("thinking")
                            params = None

                            # Handle encryption explicitly
                            if encryption_type == "master":
                                try:
                                    content = decipher(content, MASTER_KEY)
                                    if thinking:
                                        thinking = decipher(thinking, MASTER_KEY)
                                except Exception as e:
                                    logger.error(
                                        f"Error decrypting message content for chat {chat_id}: {e}"
                                    )
                                    continue
                            # If encryption_type is None or empty, content is already plain text

                            # Handle params for assistant messages
                            if msg_data["role"] == "assistant" and "params" in msg_data:
                                params_str = msg_data["params"]
                                if encryption_type == "master":
                                    try:
                                        params_str = decipher(params_str, MASTER_KEY)
                                    except Exception as e:
                                        logger.error(
                                            f"Error decrypting message params for chat {chat_id}: {e}"
                                        )
                                        continue

                                try:
                                    params = json.loads(params_str)
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Error parsing message params JSON: {e}"
                                    )
                                    params = {}

                            message_id = str(uuid.uuid4())
                            message_obj = {
                                "message_id": message_id,
                                "chat_id": chat_id,
                                "role": msg_data["role"],
                                "content": content,
                                "timestamp": timestamp,
                                "thinking": thinking,
                                "params": params,
                            }

                            messages_dict[message_id] = message_obj

                            if timestamp > last_message_time:
                                last_message_time = timestamp

                        if "LastEvaluatedKey" not in response:
                            break
                        query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

                except Exception as e:
                    logger.error(f"Error querying messages for chat {chat_id}: {e}")
                    continue

                # Match usage records to this chat using pre-indexed data
                chat_usages = []
                for message_obj in messages_dict.values():
                    message_timestamp_ms = int(
                        message_obj["timestamp"].timestamp() * 1000
                    )
                    if message_timestamp_ms in usage_by_timestamp:
                        usage_obj = usage_by_timestamp[message_timestamp_ms].copy()
                        usage_obj["chat_id"] = chat_id
                        chat_usages.append(usage_obj)

                # Only include chats that have messages
                if messages_dict:
                    chat_obj = {
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "title": title,
                        "creation_time": creation_time,
                        "last_message_at": last_message_time,
                        "messages": messages_dict,
                        "usages": chat_usages,
                    }

                    user_chats[chat_id] = chat_obj

        if user_chats:
            result[user_id] = user_chats

    logger.info(
        f"Extracted {len(result)} users with {sum(len(chats) for chats in result.values())} total chats"
    )
    return result


import asyncio
from typing import Annotated, Any
from typing_extensions import TypedDict
from functools import cache, cached_property

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from psycopg.rows import dict_row
from sqlalchemy.ext.asyncio import AsyncSession
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, computed_field

from app.config import get_settings
from app.agent.checkpoint import EncryptedAsyncPostgresSaver
from app.chat.repository import ChatRepository
from app.chat_group.models import ChatGroup  # Import to register with SQLAlchemy
from app.message.schemas import MessageMetadata, MessageRoleEnum, MessageConfig
from app.message.repository import MessageMDRepository
from app.usage_tracking.repository import UsageTrackingRepository
from app.usage_tracking.constants import MODEL_PRICING
from app.db import get_db_session_ctxmgr

settings = get_settings()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_config: dict[str, Any]


async def create_new_message(
    graph, chat_id, message
) -> tuple[MessageMetadata, MessageMetadata]:
    config = {
        "configurable": {
            "thread_id": chat_id,
            "encryption_key": MASTER_KEY,
        }
    }
    parent_state = await graph.aget_state(config=config)
    parent_cid = parent_state.config["configurable"].get("checkpoint_id", "")

    input_msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": message,
            }
        ]
    )

    await graph.ainvoke(
        input={"messages": [input_msg], "model_config": {}},
        config=config,
    )

    human_checkpoint_id = ""
    ai_checkpoint_id = ""
    async for state in graph.aget_state_history(config=config):
        # First state is state that updated the last ai message
        if not ai_checkpoint_id:
            ai_checkpoint_id = state.config["configurable"]["checkpoint_id"]
            continue
        if isinstance(state.values["messages"][-1], HumanMessage):
            human_checkpoint_id = state.config["configurable"]["checkpoint_id"]
            break

    human_metadata = MessageMetadata(
        checkpoint_id=human_checkpoint_id,
        message_id=uuid.UUID(human_checkpoint_id),
        chat_id=uuid.UUID(chat_id),
        role=MessageRoleEnum.USER,
        message_config=MessageConfig(),
        parent_checkpoint_id=parent_cid,
    )
    ai_metadata = MessageMetadata(
        checkpoint_id=ai_checkpoint_id,
        message_id=uuid.UUID(ai_checkpoint_id),
        chat_id=uuid.UUID(chat_id),
        role=MessageRoleEnum.AI,
        message_config=MessageConfig(),
        parent_checkpoint_id=human_checkpoint_id,
    )

    return human_metadata, ai_metadata


class ChatSchema(BaseModel):
    chat_id: str
    title: str
    owner_id: str
    last_message_at: datetime
    created_at: datetime
    vault_mode: bool = False
    group_id: str | None = None
    favorite: bool = False


class UsageSchema(BaseModel):
    owner_id: str
    chat_id: str
    timestamp: datetime
    model: str
    task: str
    input_tokens: int = 0
    input_cache_read: int = 0
    input_cache_create: int = 0
    output_tokens: int = 0

    @computed_field
    @cached_property
    def input_cost(self) -> float:
        return (
            MODEL_PRICING[self.model]["cache_read"] * self.input_cache_read
            + MODEL_PRICING[self.model]["cache_create"] * self.input_cache_create
            + MODEL_PRICING[self.model]["input"] * self.input_tokens
        )

    @computed_field
    @cached_property
    def output_cost(self) -> float:
        return MODEL_PRICING[self.model]["output"] * self.output_tokens


async def main(users: list, debug: bool = False):
    chat_data = extract_chat_data(users)
    chat_repo = ChatRepository()
    usage_repo = UsageTrackingRepository()
    message_repo = MessageMDRepository()

    # Create Langgraph checkpointer connection
    # Set up Langgraph checkpointer with it's own async conn pool
    async_conn_pool = AsyncConnectionPool(
        conninfo=settings.POSTGRES_URL,
        max_lifetime=600,  # 10 minutes
        max_idle=300,  # Close idle connections after 5 minute
        min_size=5,
        max_size=100,
        timeout=30,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },
    )
    await async_conn_pool.open()
    if async_conn_pool.closed:
        logger.error(f"psycopg connection to POSTGRES failed.")
        raise RuntimeError("Failed to connect to PostgreSQL")

    checkpointer = EncryptedAsyncPostgresSaver(async_conn_pool)
    await checkpointer.setup()

    for user_id, user_chats in chat_data.items():
        logger.info(f"Processing user {user_id} with {len(user_chats)} chats")
        async with get_db_session_ctxmgr() as session:
            for chat_id, chat in user_chats.items():
                # Step 1: create chat schema objects from chat data and save to db
                chat_schema = ChatSchema(
                    chat_id=chat_id,
                    title=chat["title"][:settings.MAX_TITLE_LEN],
                    owner_id="test.test@dish.com" if debug else chat["user_id"],
                    last_message_at=chat["last_message_at"].replace(tzinfo=None),
                    created_at=chat["creation_time"].replace(tzinfo=None),
                    vault_mode=False,
                    favorite=False,
                )
                chat_obj = await chat_repo.create_one(session, obj_in=chat_schema)

                # Step 2: migrate all usage stats to db
                usage_schemas = [
                    UsageSchema(
                        owner_id="test.test@dish.com" if debug else usage["user_id"],
                        chat_id=chat_id,
                        timestamp=usage["timestamp"].replace(tzinfo=None),
                        model=usage["model"],
                        task="chat",
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                    )
                    for usage in chat["usages"]
                ]
                if usage_schemas:
                    await usage_repo.create_many(session, objs_in=usage_schemas)

                # Step 3: Migrate the chat message contents by simulating a conversation
                messages = list(chat["messages"].values())
                user_messages = sorted(
                    [msg for msg in messages if msg["role"] == "user"],
                    key=lambda msg: msg["timestamp"],
                )

                assistant_messages = sorted(
                    [msg for msg in messages if msg["role"] == "assistant"],
                    key=lambda msg: msg["timestamp"],
                )

                # Fake LLM for simulating LLM execution and populating the checkpointer
                fakellm = FakeListChatModel(
                    responses=[msg["content"] for msg in assistant_messages]
                )

                def generate_response(state: AgentState, config=None):
                    """Generate the response from the current message history

                    Cachepoint the message history aggressively
                    """
                    messages = state["messages"]
                    response = fakellm.invoke(messages, config=config)
                    return {"messages": [response]}

                fakellm_graph = StateGraph(AgentState)
                fakellm_graph.add_node("agent", generate_response)
                fakellm_graph.add_edge(START, "agent")
                fakellm_graph.add_edge("agent", END)
                compiled_graph = fakellm_graph.compile(checkpointer=checkpointer)

                for m in user_messages:
                    human_md, ai_md = await create_new_message(
                        compiled_graph, chat_id, m["content"]
                    )
                    await message_repo.create_many(session, objs_in=[human_md, ai_md])

                # Update active checkpoint
                await chat_repo.update(
                    session,
                    db_obj=chat_obj,
                    obj_in={"active_checkpoint": ai_md.checkpoint_id},
                )
    await async_conn_pool.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate chat data from DynamoDB to PostgreSQL"
    )
    parser.add_argument(
        "--users",
        nargs="*",
        default=[],
        help="List of user IDs to migrate. If empty, migrates all users.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging",
    )
    args = parser.parse_args()

    users = args.users
    print(users)
    if users:
        logger.info(f"Migrating data for users: {users}")
    else:
        logger.info("Migrating data for all users")
    asyncio.run(main(users, args.debug))
