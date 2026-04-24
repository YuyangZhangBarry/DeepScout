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
    use_rag: bool | None = Field(
        default=None,
        description="None: follow settings RAG_ENABLED + keys; False: skip vector retrieval",
    )
    rag_top_k: int | None = Field(
        default=None,
        ge=1,
        le=40,
        description="Top-k chunks from Chroma for synthesis context",
    )
    use_rag_hybrid: bool | None = Field(
        default=None,
        description="None: follow RAG_HYBRID_ENABLED; False: dense-only (no BM25/RRF)",
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
    search_retry_queries: list[str] = Field(
        default_factory=list,
        description="Second-pass search queries when the first search returned no hits (empty if unused)",
    )
    evidence: list[EvidenceSourceOut]
    answer: ResearchAnswerPayload
    rag_used: bool = Field(
        default=False,
        description="True when Chroma+embeddings retrieval supplied synthesis context",
    )
    phase_trace: list[str] = Field(
        default_factory=list,
        description="Ordered phase transitions for debugging (PLANNING/TOOLING/…)",
    )


class ResearchJobCreatedResponse(BaseModel):
    job_id: str
    status: str = "pending"
    poll_url: str
    events_url: str


class ResearchJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str | None = None
    error: str | None = None
    events: list[dict] = Field(default_factory=list)
    result: ResearchResponseBody | None = None
