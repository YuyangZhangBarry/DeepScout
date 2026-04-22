#!/usr/bin/env bash
# Week 1 acceptance: one curl end-to-end; server logs must show
#   [research] stage=plan → stage=papers → (optional stage=fetch) → stage=answer
#
# Prereqs: .env with DEEPSEEK_API_KEY; SEMANTIC_SCHOLAR_API_KEY recommended.
# Run API (separate terminal), INFO logs:
#   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --log-level info
#
# Then:
#   ./scripts/week1_acceptance.sh
# Or:
#   BASE=http://127.0.0.1:8000 ./scripts/week1_acceptance.sh

set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"
BODY='{"question":"What is retrieval-augmented generation (RAG) for large language models? Give a short survey angle.","max_fetch_urls":0,"max_papers":8}'

echo "POST ${BASE}/v1/research/ ..."
curl -sS -X POST "${BASE}/v1/research/" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: week1-acceptance" \
  -d "${BODY}" \
  | python3 -m json.tool

echo ""
echo "Done. On the server terminal, grep this run:"
echo "  grep '\\[research\\]'   # plan → papers → answer"
