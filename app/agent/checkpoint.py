from typing import Optional, Any, Sequence, cast
from collections.abc import AsyncIterator
import asyncio

from psycopg import AsyncPipeline
from psycopg.types.json import Jsonb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import _ainternal
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from app.core.encryption import cipher, decipher

class Aes256Cipher:
    def encrypt(self, plaintext: bytes, b64key: str) -> tuple[str, bytes]:
        '''
        Encrypt message bytes with AES-256 in GCM mode.
        Return cyphertext prepended with nonce and tag in bytes
        '''

        ciphertext = cipher(plaintext, b64key)

        return "aes256", ciphertext

    def decrypt(self, ciphername: str, ciphertext: bytes, b64key: str) -> bytes:
        '''
        Decrypt decoded message with AES-256 in GCM mode.
        Return plaintext in bytes
        '''
        assert ciphername == "aes256", f"Unsupported cipher: {ciphername}"

        plaintext = decipher(ciphertext, b64key)

        return plaintext

class ConfigurableEncryptedSerializer(SerializerProtocol):
    """Serializer that encrypts and decrypts data using an encryption protocol."""

    def __init__(
        self, serde: SerializerProtocol = JsonPlusSerializer()
    ) -> None:
        self.cipher = Aes256Cipher()
        self.serde = serde

    def dumps(self, obj: Any) -> bytes:
        return self.serde.dumps(obj)

    def loads(self, data: bytes) -> Any:
        return self.serde.loads(data)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self.serde.dumps_typed(obj)
    
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads_typed(data)
    
    def dumps_typed_encrypted(self, obj: Any, b64key: str) -> tuple[str, bytes]:
        """Serialize an object to a tuple (type, bytes) and encrypt the bytes with the provided key."""
        # serialize data
        typ, data = self.serde.dumps_typed(obj)
        # encrypt data
        ciphername, ciphertext = self.cipher.encrypt(data, b64key)
        # add cipher name to type
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed_encrypted(self, data: tuple[str, bytes], b64key: str) -> Any:
        enc_cipher, ciphertext = data
        # unencrypted data
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)
        # extract cipher name
        typ, ciphername = enc_cipher.split("+", 1)
        # decrypt data
        decrypted_data = self.cipher.decrypt(ciphername, ciphertext, b64key)
        # deserialize data
        return self.serde.loads_typed((typ, decrypted_data))
    
class EncryptedAsyncPostgresSaver(AsyncPostgresSaver):
    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: Optional[AsyncPipeline] = None,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        super().__init__(
            conn=conn,
            pipe=pipe,
            serde=ConfigurableEncryptedSerializer()
        )
    
    def _load_checkpoint_encrypted(
        self,
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
        pending_sends: list[tuple[bytes, bytes]],
        b64key: str,
    ) -> Checkpoint:
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed_encrypted((c.decode(), b), b64key) for c, b in pending_sends or []
            ],
            "channel_values": self._load_blobs_encrypted(channel_values, b64key),
        }
    
    def _load_blobs_encrypted(
        self, blob_values: list[tuple[bytes, bytes, bytes]], b64key: str
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed_encrypted((t.decode(), v), b64key)
            for k, t, v in blob_values
            if t.decode() != "empty"
        }
    
    def _dump_blobs_encrypted(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
        b64key: str
    ) -> list[tuple[str, str, str, str, str, Optional[bytes]]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed_encrypted(values[k], b64key)
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes_encrypted(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]], b64key: str
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed_encrypted((t.decode(), v), b64key),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes_encrypted(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
        b64key: str
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed_encrypted(value, b64key),
            )
            for idx, (channel, value) in enumerate(writes)
        ]
    
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Decrypt and list checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        b64key = config["configurable"].get("encryption_key")
        if not b64key or not isinstance(b64key, str):
            raise ValueError("No valid encryption key is provided in the checkpoint config")
        
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        async with self._cursor() as cur:
            await cur.execute(query, args, binary=True)
            async for value in cur:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint_encrypted,
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                        b64key,
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": value["checkpoint_ns"],
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes_encrypted, value["pending_writes"], b64key),
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Decrypt and get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        b64key = config["configurable"].get("encryption_key")
        if not b64key or not isinstance(b64key, str):
            raise ValueError("No valid encryption key is provided in the checkpoint config")
        
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
                binary=True,
            )

            async for value in cur:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint_encrypted,
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                        b64key,
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes_encrypted, value["pending_writes"], b64key),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Encrypt and save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        b64key = config["configurable"].get("encryption_key")
        if not b64key or not isinstance(b64key, str):
            raise ValueError("No valid encryption key is provided in the checkpoint config")
        
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                await asyncio.to_thread(
                    self._dump_blobs_encrypted,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),  # type: ignore[misc]
                    new_versions,
                    b64key,
                ),
            )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(self._dump_checkpoint(copy)),
                    self._dump_metadata(get_checkpoint_metadata(config, metadata)),
                ),
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Encrypt and store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        b64key = config["configurable"].get("encryption_key")
        if not b64key or not isinstance(b64key, str):
            raise ValueError("No valid encryption key is provided in the checkpoint config")
        
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes_encrypted,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
            b64key,
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(query, params)