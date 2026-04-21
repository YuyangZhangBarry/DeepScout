from pydantic import BaseModel, Field


class ResearchRequestBody(BaseModel):
    question: str = Field(..., min_length=4, max_length=8000)
    max_subqueries: int | None = Field(
        default=None,
        ge=3,
        le=12,
        description="LLM planning: number of Semantic Scholar queries (uses settings default if omitted)",
    )
    max_papers: int | None = Field(
        default=None,
        ge=1,
        le=40,
        description="Cap merged search hits before fetch/synthesis",
    )
    max_fetch_urls: int | None = Field(
        default=None,
        ge=0,
        le=15,
        description="Fetch full HTML for first N hits; 0 = abstracts/snippets only",
    )


class KeyPoint(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class CitationEntry(BaseModel):
    source_id: str
    paper_id: str | None = None
    url: str
    title: str = ""


class ResearchAnswerPayload(BaseModel):
    executive_summary: str
    key_points: list[KeyPoint] = Field(default_factory=list)
    limitations: str = ""
    report_markdown: str = ""
    citations: list[CitationEntry] = Field(default_factory=list)


class EvidenceSourceOut(BaseModel):
    source_id: str
    paper_id: str | None = None
    url: str
    title: str = ""
    excerpt: str = ""


class ResearchResponseBody(BaseModel):
    question: str
    planning_queries: list[str]
    evidence: list[EvidenceSourceOut]
    answer: ResearchAnswerPayload
