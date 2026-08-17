from pydantic import BaseModel

class EmbeddedPassage(BaseModel):
    text: str
    embedding: list[float]
    metadata: dict

class EmbeddedDocument(BaseModel):
    name: str
    est_size: int
    summary: str = ""
    keywords: list[str] = []
    passages: list[EmbeddedPassage]