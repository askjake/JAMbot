import logging
import asyncio

from typing import Any, Generic, Type, TypeVar, Union, Dict, List, Sequence
from pydantic import BaseModel
from sqlalchemy import select, inspect, delete as sqlalchemy_delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import UnmappedClassError
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import Base  # Assuming your Base is here (from DeclarativeBase)
from app.core.utils import get_utc_now_notz

# Define TypeVars for generic repository
# ModelType represents the SQLAlchemy model (e.g., Foo, User)
ModelType = TypeVar("ModelType", bound=Base)
# CreateSchemaType represents the Pydantic schema for creating an object
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
# UpdateSchemaType represents the Pydantic schema for updating an object
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

logger = logging.getLogger(__name__)


# Helper functions
def _get_primary_key_column(model_cls: Type) -> Any | List[Any] | None:
    """Gets the primary key Column object for a model."""
    try:
        mapper = inspect(model_cls).mapper
        pk_columns = mapper.primary_key
        if len(pk_columns) == 1:
            return pk_columns[0]
        elif len(pk_columns) == 0:
            logger.error(f"Model {model_cls.__name__} has no primary key.")
            return None
        else:
            logger.error(
                f"Model {model_cls.__name__} has a composite primary key. "
                "get_by_ids currently only supports single primary keys."
            )
            return None
    except UnmappedClassError:
        logger.error(f"Class {model_cls.__name__} is not a mapped SQLAlchemy class.")
        return None
    except Exception as e:
        logger.error(
            f"Error inspecting primary key for {model_cls.__name__}: {e}", exc_info=True
        )
        return None


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic base repository providing common CRUD operations for SQLAlchemy models.

    Parameters:
        model: The SQLAlchemy model class.
    """

    def __init__(self, model: Type[ModelType]):
        self.model = model
        # Store the PK info
        self.pk_column = _get_primary_key_column(self.model)
        if self.pk_column is None:
            raise ValueError(
                f"BaseRepository requires model {model.__name__} to have a single primary key column."
            )

    def _handle_db_error(self, e: SQLAlchemyError, context: str = "database operation"):
        """Handles common SQLAlchemy errors."""
        logger.error(f"Database error during {context}: {e}", exc_info=True)
        # You might raise a custom exception here, e.g., raise DatabaseError(str(e))
        # For simplicity, we re-raise the original error or just log it
        # Depending on your error handling strategy across the app
        raise e  # Re-raising allows higher layers (like services or endpoints) to catch

    async def get_one_by_id(self, db: AsyncSession, id: Any) -> ModelType | None:
        """
        Get a single object by its primary key.

        Args:
            db: The database session.
            id: The primary key value.

        Returns:
            The object instance or None if not found.
        """
        try:
            # session.get() is efficient for primary key lookups
            result = await db.get(self.model, id)
            return result
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"get by id {id}")
        except Exception as e:  # Catch unexpected errors
            logger.error(f"Unexpected error during get by id {id}: {e}", exc_info=True)
            raise e

    async def get_many_by_ids(
        self, db: AsyncSession, ids: Sequence[Any]
    ) -> List[ModelType]:
        """
        Get a list of objects by their primary keys.

        Args:
            db: The database session.
            ids: The primary key values.

        Returns:
            A list of object instances found
        """
        if not ids:
            return []

        try:
            q = select(self.model).where(self.pk_column.in_(ids))
            rows = await db.execute(q)
            objects = rows.scalars().all()
            return list(objects)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"get_by_ids for model {self.model.__name__}")
        except Exception as e:
            logger.error(
                f"Unexpected error during get_by_ids for model {self.model.__name__}: {e}",
                exc_info=True,
            )
            raise e

        return []

    async def create_one(
        self, db: AsyncSession, *, obj_in: CreateSchemaType
    ) -> ModelType:
        """
        Create a new object in the database.

        Args:
            db: The database session.
            obj_in: The Pydantic schema containing the data for the new object.

        Returns:
            The newly created object instance.
        """
        try:
            # Create SQLAlchemy model instance from Pydantic schema
            # Exclude unset ensures optional fields with default values aren't overwritten by None
            obj_in_data = obj_in.model_dump(exclude_unset=True)
            db_obj = self.model(**obj_in_data)
            db.add(db_obj)
            await db.flush()  # Flush to assign generated values like IDs, but don't commit yet
            await db.refresh(db_obj)  # Refresh to get the updated state from the DB
            return db_obj
        except SQLAlchemyError as e:
            # Rollback is typically handled by the session context manager/dependency
            self._handle_db_error(e, f"create object data={obj_in}")
        except Exception as e:
            logger.error(f"Unexpected error during create: {e}", exc_info=True)
            raise e

    async def create_many(
        self, db: AsyncSession, *, objs_in: Sequence[CreateSchemaType]
    ) -> List[ModelType]:
        """
        Create a list of new objects in the database.

        Args:
            db: The database session.
            objs_in: A list of Pydantic schemas containing the data for the new object.

        Returns:
            A list of newly created object instances.
        """
        try:
            # Create SQLAlchemy model instance from Pydantic schema
            # Exclude unset ensures optional fields with default values aren't overwritten by None
            objs_in_data = [obj_in.model_dump(exclude_unset=True) for obj_in in objs_in]
            db_objs = [self.model(**data) for data in objs_in_data]
            db.add_all(db_objs)
            await db.flush()  # Flush to assign generated values like IDs, but don't commit yet

            task_refreshes = [db.refresh(db_obj) for db_obj in db_objs]
            await asyncio.gather(*task_refreshes)

            return db_objs
        except SQLAlchemyError as e:
            # Rollback is typically handled by the session context manager/dependency
            self._handle_db_error(e, f"create object data={objs_in}")
        except Exception as e:
            logger.error(f"Unexpected error during create: {e}", exc_info=True)
            raise e

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,  # The existing SQLAlchemy object instance
        obj_in: Union[
            UpdateSchemaType, Dict[str, Any]
        ],  # New data (Pydantic schema or dict)
    ) -> ModelType:
        """
        Update an existing object in the database.

        Args:
            db: The database session.
            db_obj: The existing SQLAlchemy object instance to update.
            obj_in: A Pydantic schema or dictionary containing the fields to update.

        Returns:
            The updated object instance.
        """
        try:
            if isinstance(obj_in, dict):
                update_data = obj_in
            else:  # It's a Pydantic model
                update_data = obj_in.model_dump(exclude_unset=True)

            if not update_data:  # No fields were provided to update
                return db_obj

            # Refresh update timestamp if exists
            update_data["updated_at"] = get_utc_now_notz()

            # Update the model instance fields
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
                else:
                    # Optional: Log or raise error if field doesn't exist
                    logger.warning(
                        f"Attempted to update non-existent field '{field}' on model {self.model.__name__}"
                    )

            db.add(db_obj)  # Add the modified object to the session
            await db.flush()  # Flush changes to the DB transaction
            await db.refresh(db_obj)  # Refresh to get the updated state
            return db_obj
        except SQLAlchemyError as e:
            # Rollback is typically handled by the session context manager/dependency
            self._handle_db_error(
                e, f"update object id={getattr(db_obj, 'id', 'N/A')} data={obj_in}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during update: {e}", exc_info=True)
            raise e

    async def remove_by_id(self, db: AsyncSession, *, id: Any) -> ModelType | None:
        """
        Remove an object from the database by its primary key.

        Args:
            db: The database session.
            id: The primary key of the object to remove.

        Returns:
            The removed object instance or None if not found.
        """
        try:
            obj = await self.get_one_by_id(db, id)  # First, get the object
            if obj:
                await db.delete(obj)
                await db.flush()  # Flush changes to the DB transaction
            return obj  # Return the object that was deleted (or None)
        except SQLAlchemyError as e:
            # Rollback is typically handled by the session context manager/dependency
            self._handle_db_error(e, f"remove object id={id}")
        except Exception as e:
            logger.error(f"Unexpected error during remove: {e}", exc_info=True)
            raise e

    async def remove_many_by_ids(self, db: AsyncSession, *, ids: Sequence[Any]) -> int:
        """
        Remove multiple objects from the database by their IDs.

        Args:
            db: The database session.
            ids: The primary keys of objects to remove.

        Returns:
            Number of objects deleted.
        """
        if not ids:
            return 0

        try:
            # Create an efficient bulk delete query
            stmt = (
                sqlalchemy_delete(self.model)
                .where(self.pk_column.in_(ids))
                .execution_options(synchronize_session=False)
            )

            result = await db.execute(stmt)
            await db.flush()

            # Return the count of deleted rows
            return result.rowcount

        except SQLAlchemyError as e:
            self._handle_db_error(e, f"remove_many for ids={ids}")
        except Exception as e:
            logger.error(f"Unexpected error during remove_many: {e}", exc_info=True)
            raise e
