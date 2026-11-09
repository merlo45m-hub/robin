#!/usr/bin/env python3
"""Robin OSINT REST API — FastAPI wrapper for the dark-web intelligence pipeline.

Endpoints:
  GET  /health              Status, Tor connectivity, models, presets (unauthenticated)
  POST /investigate         Full pipeline (refine -> search -> filter -> scrape -> summarize)
  POST /search              Dark-web search only (raw onion results)
  POST /scrape              Scrape a specific URL via Tor or clearweb
  GET  /investigations      List all saved investigations
  GET  /investigations/{id} Fetch full investigation JSON by ID
  GET  /status              Full environment status (Tor proxy, models, presets, count)

Authentication:
  Bearer token via JOB_SEARCH_API_KEY pattern.
  Reads ROBIN_API_KEY from /etc/robin-api.env, /root/robin/robin-api.env, or environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# -----------------------------------------------------------------------------
# Module Setup & Path Resolution
# -----------------------------------------------------------------------------

ROBIN_DIR = Path(__file__).resolve().parent
if str(ROBIN_DIR) not in sys.path:
    sys.path.insert(0, str(ROBIN_DIR))

# Append venv site-packages if available
VENV_SITE_PACKAGES = ROBIN_DIR / "venv" / "lib" / "python3.12" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

# -----------------------------------------------------------------------------
# Environment & Secret Loading
# -----------------------------------------------------------------------------

ENV_FILES = [
    "/etc/robin-api.env",
    "/root/.hermes/.env",
    str(ROBIN_DIR / "robin-api.env"),
    str(ROBIN_DIR / ".env"),
    "/opt/robin/.env",
]

for env_path in ENV_FILES:
    if os.path.exists(env_path):
        try:
            load_dotenv(env_path, override=False)
        except Exception:
            pass

# Import Robin modules
try:
    import config
    # Ensure OPENROUTER_API_KEY in config reflects loaded environment
    if os.getenv("OPENROUTER_API_KEY") and not config.OPENROUTER_API_KEY:
        config.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("OPENROUTER_BASE_URL") and not config.OPENROUTER_BASE_URL:
        config.OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

    import health
    import llm
    import llm_utils
    import scrape
    import search
except ImportError as err:
    print(f"FATAL: Robin module import error: {err}", file=sys.stderr)
    raise

INVESTIGATIONS_DIR = Path(
    os.getenv("ROBIN_INVESTIGATIONS_DIR", str(ROBIN_DIR / "investigations"))
)
INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("robin_api")


def get_api_key() -> str:
    """Read ROBIN_API_KEY with priority: /etc/robin-api.env -> robin-api.env -> os.environ."""
    candidates = ["/etc/robin-api.env", str(ROBIN_DIR / "robin-api.env"), str(ROBIN_DIR / ".env")]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("ROBIN_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return os.getenv("ROBIN_API_KEY", "").strip()


# -----------------------------------------------------------------------------
# Lifespan & Application Setup
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Output expected by caller/test harness on startup
    print("ROBIN_API_READY", flush=True)
    logger.info("ROBIN_API_READY: Robin OSINT REST API listening.")
    yield


app = FastAPI(
    title="Robin Dark-Web OSINT API",
    version="1.0.0",
    description="REST API for Robin OSINT pipeline: search, scrape, filter, and summarize dark-web intelligence.",
    lifespan=lifespan,
)

# CORS enabled for Android app (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Bearer Token Auth Middleware (JOB_SEARCH_API_KEY pattern)
class ApiGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Allow health checks and OpenAPI docs without authentication
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        expected_key = get_api_key()
        if not expected_key:
            return JSONResponse(
                {"error": "unauthorized", "message": "ROBIN_API_KEY is not configured on the server."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        auth_header = request.headers.get("authorization", "")
        supplied = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else ""

        if not supplied:
            return JSONResponse(
                {"error": "unauthorized", "message": "Invalid or missing Bearer token."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 1. Try static API key (backward compat)
        if expected_key and secrets.compare_digest(supplied, expected_key):
            return await call_next(request)

        # 2. Try Supabase JWT validation
        supabase_url = os.getenv("SUPABASE_URL", "http://supabase-envoy:8000")
        supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{supabase_url}/auth/v1/user",
                headers={"Authorization": f"Bearer {supplied}", "apikey": supabase_anon_key},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return await call_next(request)
        except Exception:
            pass

        return JSONResponse(
            {"error": "unauthorized", "message": "Invalid or missing Bearer token."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


app.add_middleware(ApiGuardMiddleware)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class InvestigateRequest(BaseModel):
    query: str = Field(..., description="Target search/investigation query")
    preset: Optional[str] = Field("threat_intel", description="Preset prompt template")
    max_results: Optional[int] = Field(10, description="Max raw search results to gather")
    model: Optional[str] = Field(None, description="LLM model identifier")
    lang: Optional[str] = Field("en", description="Target language")
    timeout: Optional[int] = Field(300, description="Overall timeout in seconds")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query to submit to onion search engines")
    max_results: Optional[int] = Field(10, description="Max results to return")
    engines: Optional[List[str]] = Field(None, description="Optional list of engine names to query")
    timeout: Optional[int] = Field(60, description="Overall timeout in seconds")


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape (onion or clearweb)")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")


# -----------------------------------------------------------------------------
# Pipeline Helpers (Thread-safe synchronous functions)
# -----------------------------------------------------------------------------

def save_investigation_record(data: dict) -> tuple[str, str, str]:
    """Save an investigation dictionary to JSON in INVESTIGATIONS_DIR."""
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
    investigation_id = f"investigation_{timestamp_str}"
    filename = f"{investigation_id}.json"
    filepath = str(INVESTIGATIONS_DIR / filename)

    record = {
        "timestamp": ts.isoformat(),
        "query": data["query"],
        "refined_query": data["refined_query"],
        "model": data["model"],
        "preset": data["preset"],
        "sources": data["sources"],
        "summary": data["summary"],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return investigation_id, filepath, record["timestamp"]


def execute_pipeline(
    query: str,
    preset: str = "threat_intel",
    model: Optional[str] = None,
    max_results: Optional[int] = 10,
    lang: str = "en",
) -> dict:
    """Full Robin pipeline execution: refine -> search -> filter -> scrape -> summarize."""
    # 1. Preset validation
    if not hasattr(llm, "PRESET_PROMPTS"):
        raise RuntimeError("PRESET_PROMPTS not found in llm module.")
    if preset not in llm.PRESET_PROMPTS:
        valid = ", ".join(llm.PRESET_PROMPTS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}")

    # 2. Model resolution
    model_choices = llm_utils.get_model_choices()
    if model is not None:
        matching = [m for m in model_choices if m.lower() == model.lower()]
        if not matching:
            available = ", ".join(model_choices) if model_choices else "none"
            raise ValueError(f"Model '{model}' not available. Available models: {available}")
        selected_model = matching[0]
    else:
        if not model_choices:
            raise RuntimeError(
                "No LLM models available. Configure OPENROUTER_API_KEY in /root/.hermes/.env or /etc/robin-api.env"
            )
        selected_model = model_choices[0]

    # 3. Execution (redirect stdout to stderr so terminal noise doesn't interfere)
    with contextlib.redirect_stdout(sys.stderr):
        llm_instance = llm.get_llm(selected_model)
        refined_query = llm.refine_query(llm_instance, query)
        search_query = refined_query.replace(" ", "+")

        raw_results = search.get_search_results(search_query)
        if max_results is not None and max_results > 0:
            raw_results = raw_results[:max_results]

        filtered_results = llm.filter_results(llm_instance, refined_query, raw_results)
        scraped_content = scrape.scrape_multiple(filtered_results)
        summary = llm.generate_summary(
            llm_instance,
            query,
            scraped_content,
            preset=preset,
        )

    # 4. Save to investigations directory
    inv_id, _, ts_iso = save_investigation_record({
        "query": query,
        "refined_query": refined_query,
        "model": selected_model,
        "preset": preset,
        "sources": filtered_results,
        "summary": summary,
    })

    return {
        "query": query,
        "refined_query": refined_query,
        "results": filtered_results,
        "summary": summary,
        "investigation_id": inv_id,
        "timestamp": ts_iso,
    }


def execute_search_only(
    query: str,
    max_results: Optional[int] = 10,
    engines: Optional[List[str]] = None,
) -> list:
    """Execute search across Tor onion search engines without scraping or summarizing."""
    search_query = query.replace(" ", "+")
    if engines:
        engine_lower = [e.lower() for e in engines]
        target_endpoints = [
            e["url"] for e in search.SEARCH_ENGINES
            if e["name"].lower() in engine_lower or any(el in e["name"].lower() for el in engine_lower)
        ]
        if not target_endpoints:
            target_endpoints = search.DEFAULT_SEARCH_ENGINES
    else:
        target_endpoints = search.DEFAULT_SEARCH_ENGINES

    results = []
    with ThreadPoolExecutor(max_workers=min(len(target_endpoints), 8)) as executor:
        futures = [executor.submit(search.fetch_search_results, ep, search_query) for ep in target_endpoints]
        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    results.extend(res)
            except Exception:
                pass

    seen_links = set()
    unique_results = []
    for res in results:
        link = res.get("link") or ""
        try:
            clean_link = search._dedup_key(link)
        except Exception:
            clean_link = link.rstrip("/")
        if clean_link not in seen_links:
            seen_links.add(clean_link)
            unique_results.append(res)

    if max_results is not None and max_results > 0:
        unique_results = unique_results[:max_results]
    return unique_results


def execute_scrape_url(url: str, timeout: int = 30) -> tuple[str, bool, Optional[str]]:
    """Scrape a single URL using Tor for .onion addresses or standard session for clearweb."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "", False, f"Invalid scheme '{parsed.scheme}': must be http or https"

    use_tor = (parsed.hostname or "").lower().endswith(".onion")
    session = scrape._get_session(use_tor=use_tor)
    headers = {
        "User-Agent": random.choice(scrape.USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }

    try:
        resp = session.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code == 200:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type and not any(t in content_type for t in scrape.ALLOWED_CONTENT_TYPES):
                return "", False, f"Unsupported Content-Type: {content_type}"

            chunks = []
            bytes_read = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                bytes_read += len(chunk)
                if bytes_read > scrape.MAX_DOWNLOAD_BYTES:
                    break
                chunks.append(chunk)

            html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = soup.get_text(separator=" ")
            text = " ".join(text.split())[:scrape.MAX_EXTRACTED_TEXT_CHARS]
            return text, True, None
        else:
            return "", False, f"HTTP status {resp.status_code}"
    except Exception as exc:
        return "", False, str(exc)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Endpoint 1: GET /health — return status, tor connectivity, models, presets."""
    tor_check = health.check_tor_proxy()
    tor_ok = bool(tor_check.get("status") == "up")
    models = llm_utils.get_model_choices()
    presets = list(getattr(llm, "PRESET_PROMPTS", {}).keys())

    return {
        "status": "ok",
        "tor": tor_ok,
        "models": models,
        "presets": presets,
    }


@app.post("/investigate")
async def investigate(req: InvestigateRequest):
    """Endpoint 2: POST /investigate — run full Robin pipeline and save results."""
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    timeout_seconds = req.timeout if req.timeout and req.timeout > 0 else 300
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                execute_pipeline,
                query=clean_query,
                preset=req.preset or "threat_intel",
                model=req.model,
                max_results=req.max_results,
                lang=req.lang or "en",
            ),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Investigation timed out after {timeout_seconds}s.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.exception("Investigation pipeline failure")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    """Endpoint 3: POST /search — raw search results only (no scrape/summarize)."""
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    timeout_seconds = req.timeout if req.timeout and req.timeout > 0 else 60
    try:
        results = await asyncio.wait_for(
            run_in_threadpool(
                execute_search_only,
                query=clean_query,
                max_results=req.max_results,
                engines=req.engines,
            ),
            timeout=timeout_seconds,
        )
        return {
            "query": clean_query,
            "count": len(results),
            "results": results,
        }
    except asyncio.TimeoutError:
        # Previously this endpoint had no timeout at all, so a slow engine set could
        # hold a worker (and the caller's connection) open indefinitely.
        raise HTTPException(
            status_code=504,
            detail=(
                f"Search timed out after {timeout_seconds}s. The search engines did not "
                "respond in time - try again, or lower max_results."
            ),
        )
    except Exception as e:
        logger.exception("Search execution error")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/scrape")
async def scrape_endpoint(req: ScrapeRequest):
    """Endpoint 4: POST /scrape — scrape a specific URL and return text content."""
    clean_url = req.url.strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    timeout_seconds = req.timeout if req.timeout and req.timeout > 0 else 30
    try:
        content, success, error_msg = await run_in_threadpool(
            execute_scrape_url,
            url=clean_url,
            timeout=timeout_seconds,
        )
        resp = {
            "url": clean_url,
            "content": content,
            "success": success,
        }
        if error_msg:
            resp["error"] = error_msg
        return resp
    except Exception as e:
        logger.exception("Scraping error")
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")


@app.get("/investigations")
async def list_investigations():
    """Endpoint 5: GET /investigations — list past investigations (id, query, timestamp, count)."""
    items = []
    if INVESTIGATIONS_DIR.exists():
        for fname in sorted(os.listdir(INVESTIGATIONS_DIR), reverse=True):
            if fname.endswith(".json"):
                fpath = INVESTIGATIONS_DIR / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    inv_id = fname[:-5]
                    sources = data.get("sources") or data.get("results") or []
                    items.append({
                        "id": inv_id,
                        "query": data.get("query", ""),
                        "timestamp": data.get("timestamp", ""),
                        "result_count": len(sources),
                    })
                except Exception:
                    continue

    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return items


@app.get("/investigations/{investigation_id}")
async def get_investigation_by_id(investigation_id: str):
    """Endpoint 6: GET /investigations/{id} — get full investigation record by ID."""
    safe_name = os.path.basename(investigation_id)
    candidates = [
        INVESTIGATIONS_DIR / safe_name,
        INVESTIGATIONS_DIR / f"{safe_name}.json",
    ]
    if not safe_name.startswith("investigation_"):
        candidates.extend([
            INVESTIGATIONS_DIR / f"investigation_{safe_name}",
            INVESTIGATIONS_DIR / f"investigation_{safe_name}.json",
        ])

    target_file = None
    for cand in candidates:
        if cand.is_file():
            target_file = cand
            break

    if not target_file:
        raise HTTPException(
            status_code=404,
            detail=f"Investigation '{investigation_id}' not found.",
        )

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # The filename IS the investigation id and the stored record does not carry
        # one. Emit both keys: the Android model maps `investigation_id`, while the
        # rest of the API uses `id`.
        data.setdefault("id", target_file.stem)
        data.setdefault("investigation_id", target_file.stem)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read investigation file: {e}")


@app.get("/investigations/{investigation_id}/similar")
async def find_similar_investigations(investigation_id: str, limit: int = 5):
    """Find similar past investigations using Supabase vector search."""
    import urllib.request as _ur
    safe_name = os.path.basename(investigation_id)
    candidates = [INVESTIGATIONS_DIR / safe_name, INVESTIGATIONS_DIR / f"{safe_name}.json"]
    if not safe_name.startswith("investigation_"):
        candidates.extend([INVESTIGATIONS_DIR / f"investigation_{safe_name}", INVESTIGATIONS_DIR / f"investigation_{safe_name}.json"])
    target = None
    for c in candidates:
        if c.is_file():
            target = c
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Investigation '{investigation_id}' not found.")
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read: {e}")
    # Get summary text for embedding query
    summary = data.get("summary", "") or data.get("query", "")
    if not summary:
        raise HTTPException(status_code=400, detail="No summary text to search with.")
    # Call Supabase match_investigations RPC
    supabase_url = os.getenv("SUPABASE_URL", "http://127.0.0.1:8000")
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    # We don't have an embedding yet, so return empty results with a note
    # The Android app should use the Supabase vector_search MCP directly
    return {"investigation_id": investigation_id, "query": summary[:200], "similar": [], "note": "Use Supabase vector_search MCP for semantic similarity. Embedding generation requires fastembed on the client."}

@app.get("/status")
async def get_full_status():
    """Endpoint 7: GET /status — full environment status."""
    tor_info = health.check_tor_proxy()
    models = llm_utils.get_model_choices()
    presets = list(getattr(llm, "PRESET_PROMPTS", {}).keys())

    inv_count = 0
    if INVESTIGATIONS_DIR.exists():
        inv_count = len([f for f in os.listdir(INVESTIGATIONS_DIR) if f.endswith(".json")])

    return {
        "status": "ok",
        "tor": tor_info,
        "models": models,
        "presets": presets,
        "investigations_count": inv_count,
        "tor_proxy": "127.0.0.1:9050",
        "investigations_dir": str(INVESTIGATIONS_DIR),
        "server_time": datetime.now().isoformat(),
    }


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8084"))
    logger.info("Starting Robin REST API on %s:%d ...", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=300,
        log_level="info",
    )
