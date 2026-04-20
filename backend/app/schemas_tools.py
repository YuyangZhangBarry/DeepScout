from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source_query: str = ""


class SearchRequestBody(BaseModel):
    queries: list[str] = Field(..., min_length=1, description="Parallel web search queries")
    max_results_per_query: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override default from settings when set",
    )


class SearchResponseBody(BaseModel):
    items: list[SearchHit]


class FetchRequestBody(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=50)


class FetchedDocument(BaseModel):
    url: str
    final_url: str = ""
    title: str = ""
    text: str = ""
    content_hash: str = ""
    status_code: int = 0
    error: str | None = None


class FetchResponseBody(BaseModel):
    documents: list[FetchedDocument]
