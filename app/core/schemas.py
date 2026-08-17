from typing import Generic, TypeVar, List
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)
class Pagination(BaseModel,  Generic[T]):
    docs: List[T]
    totalDocs: int
    limit: int
    page: int
    totalPages: int 
    hasNextPage: bool
    nextPage: int | None
    hasPrevPage: bool
    prevPage: int | None