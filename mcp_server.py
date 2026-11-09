#!/usr/bin/env python3
"""Robin MCP server — exposes the dark-web OSINT pipeline as MCP tools over stdio.

Tools:
  robin_investigate  Run the full pipeline (refine -> search -> filter -> scrape
                     -> summarize) for one query and return summary + sources.
  robin_status       Report available models, valid presets, and environment
                     health (Tor proxy, robin dir, investigation output dir).

The pipeline steps mirror robin_cli.main() exactly (same call order, same
arguments, same save format); the on-disk JSON stays byte-compatible with the
CLI's so MCP and CLI investigations are interchangeable downstream.

Wire safety: stdout is the JSON-RPC channel. Every step of the pipeline is
executed with stdout redirected to stderr (robin's llm/search modules print
progress and warnings), and the pipeline runs in a worker thread so the event
loop keeps serving.

Env: ROBIN_INVESTIGATIONS_DIR overrides where investigation JSON is written.

Run with the interpreter that has robin's dependencies (mcp, bs4, langchain-*,
requests, PySocks). On this box that is /data/data/com.termux/files/usr/bin/python3;
the Hermes gateway venv lacks bs4/langchain, so launching this server with the
gateway python starts fine but every pipeline call fails at import. Verify with:

    <python> -c "import mcp, bs4, langchain_ollama, socks"

Run:  python3 /data/data/com.termux/files/home/robin/mcp_server.py
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

ROBIN_DIR = Path(__file__).resolve().parent
INVESTIGATIONS_DIR = Path(
    os.environ.get("ROBIN_INVESTIGATIONS_DIR", str(ROBIN_DIR / "investigations"))
)
SERVER_NAME = "robin"
SERVER_VERSION = "1.0.0"

# robin's modules use bare `import config` / `import search`; make sure the
# checkout is importable no matter what cwd the MCP client launches us from.
if str(ROBIN_DIR) not in sys.path:
    sys.path.insert(0, str(ROBIN_DIR))


def _log(message: str) -> None:
    """Diagnostics go to stderr — never stdout, which carries the protocol."""
    print(f"[robin-mcp] {message}", file=sys.stderr, flush=True)


def save_investigation(data: dict[str, Any]) -> tuple[str, str]:
    """Persist an investigation to ``investigations/investigation_<ts>.json``.

    Byte-compatible with ``robin_cli.save_investigation`` (same directory, same
    filename format, same key set and indent) so CLI and MCP runs are
    interchangeable downstream.

    robin_cli itself is deliberately NOT imported: its module body calls
    ``sys.exit(1)`` when a robin dependency is missing, and a SystemExit raised
    inside a worker thread takes the whole MCP server down with it.
    """
    investigations_dir = INVESTIGATIONS_DIR
    investigations_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filepath = investigations_dir / f"investigation_{timestamp_str}.json"

    save_data = {
        "timestamp": timestamp.isoformat(),
        "query": data["query"],
        "refined_query": data["refined_query"],
        "model": data["model"],
        "preset": data["preset"],
        "sources": data["sources"],
        "summary": data["summary"],
    }

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(save_data, handle, indent=2)

    return str(filepath), save_data["timestamp"]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def _load_pipeline_modules():
    """Import the robin modules lazily so the server still starts (and can
    report a clean error) when optional deps such as langchain are missing.

    SystemExit/KeyboardInterrupt are converted to RuntimeError: robin modules
    (and robin_cli) call ``sys.exit(1)`` on import failure, and an uncaught
    SystemExit escaping a worker thread tears down the whole MCP server.
    """
    try:
        import llm
        import llm_utils
        import scrape
        import search
    except BaseException as exc:  # noqa: BLE001 - never let SystemExit escape
        raise RuntimeError(
            f"could not import robin pipeline modules: {type(exc).__name__}: {exc}"
        ) from exc

    return llm, llm_utils, scrape, search


def resolve_model(llm_utils: Any, requested: str | None) -> str:
    """Pick the model exactly as robin_cli does: explicit flag validated against
    get_model_choices(), otherwise the first available model."""
    model_choices = llm_utils.get_model_choices()
    if not model_choices:
        raise RuntimeError(
            "No LLM models available. Configure at least one API key in the "
            "robin .env file (OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            "GOOGLE_API_KEY / OPENROUTER_API_KEY, or a local Ollama/llama.cpp/"
            "custom endpoint) and restart the MCP server."
        )
    if requested:
        if requested not in model_choices:
            raise ValueError(
                f"Model '{requested}' not available. Available models: "
                + ", ".join(model_choices)
            )
        return requested
    return model_choices[0]


def validate_preset(llm: Any, preset: str) -> str:
    if not hasattr(llm, "PRESET_PROMPTS"):
        raise RuntimeError("PRESET_PROMPTS not found in the robin llm module.")
    if preset not in llm.PRESET_PROMPTS:
        raise ValueError(
            f"Unknown preset: {preset}. Valid presets: "
            + ", ".join(llm.PRESET_PROMPTS.keys())
        )
    return preset


def _make_notifier(progress: Any) -> Any:
    """Normalize a progress callback to ``(message, step)``.

    Accepts either ``fn(message)`` or ``fn(message, step)`` so callers that
    don't care about step numbers still work. The arity is decided by
    inspection, not by catching TypeError — a TypeError raised inside the
    callback must not be swallowed and retried.
    """
    if progress is None:
        return lambda _msg, _step=None: None

    try:
        params = [
            p
            for p in inspect.signature(progress).parameters.values()
            if p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]
        takes_step = any(p.kind is p.VAR_POSITIONAL for p in params) or len(params) >= 2
    except (TypeError, ValueError):  # builtins / C callables: assume two-arg
        takes_step = True

    if takes_step:
        return lambda message, step=None: progress(message, step)
    return lambda message, step=None: progress(message)


def run_pipeline(
    query: str,
    preset: str = "threat_intel",
    model: str | None = None,
    max_results: int | None = None,
    save: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Blocking full-pipeline run. Call via asyncio.to_thread from the server.

    Returns a dict: query, refined_query, model, preset, summary, sources,
    num_sources, file, timestamp, elapsed_seconds.
    """
    notify = _make_notifier(progress)
    started = time.time()

    llm, llm_utils, scrape, search = _load_pipeline_modules()

    query = (query or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")

    selected_model = resolve_model(llm_utils, model)
    preset = validate_preset(llm, preset)
    notify(f"model={selected_model} preset={preset}", 1)

    # robin_cli redirects stdout for the whole pipeline: robin's modules print
    # progress and rate-limit notices, and stdout is our protocol channel.
    with contextlib.redirect_stdout(sys.stderr):
        llm_instance = llm.get_llm(selected_model)

        notify("refining query", 2)
        refined_query = llm.refine_query(llm_instance, query)
        search_query = refined_query.replace(" ", "+")
        notify(f"refined query: {refined_query!r}", 2)

        notify("searching dark-web engines", 3)
        results = search.get_search_results(search_query)

        if max_results is not None:
            results = results[:max_results]
        notify(f"{len(results)} raw results", 3)

        notify("filtering results with LLM", 4)
        filtered_results = llm.filter_results(llm_instance, refined_query, results)
        notify(f"{len(filtered_results)} filtered sources", 4)

        notify("scraping sources", 5)
        scraped_content = scrape.scrape_multiple(filtered_results)

        notify("generating summary", 6)
        summary = llm.generate_summary(
            llm_instance, query, scraped_content, preset=preset
        )

        investigation_data = {
            "query": query,
            "refined_query": refined_query,
            "model": selected_model,
            "preset": preset,
            "sources": filtered_results,
            "summary": summary,
        }

        filepath = None
        timestamp_iso = None
        if save:
            filepath, timestamp_iso = save_investigation(investigation_data)

    return {
        "query": query,
        "refined_query": refined_query,
        "model": selected_model,
        "preset": preset,
        "summary": summary,
        "sources": filtered_results,
        "num_sources": len(filtered_results),
        "file": filepath,
        "timestamp": timestamp_iso,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def render_result(result: dict[str, Any]) -> str:
    """Human/LLM-readable text block: summary followed by the source list."""
    lines = [
        f"Query: {result['query']}",
        f"Refined query: {result['refined_query']}",
        f"Model: {result['model']} | Preset: {result['preset']} | "
        f"Sources: {result['num_sources']} | {result['elapsed_seconds']}s",
    ]
    if result.get("file"):
        lines.append(f"Saved: {result['file']}")
    lines += ["", "=== INVESTIGATION SUMMARY ===", result["summary"] or "(empty)"]

    sources = result.get("sources") or []
    lines += ["", f"=== SOURCES ({len(sources)}) ==="]
    if sources:
        for i, src in enumerate(sources, 1):
            title = (src.get("title") or "Untitled").strip()
            link = (src.get("link") or "").strip()
            lines.append(f"{i}. {title} — {link}")
    else:
        lines.append("(no sources returned)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Environment status
# --------------------------------------------------------------------------- #

def check_tor_proxy(host: str = "127.0.0.1", port: int = 9050, timeout: float = 5.0):
    """Mirror health.check_tor_proxy: is the SOCKS proxy accepting connections?"""
    try:
        start = time.time()
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return {
            "status": "up",
            "endpoint": f"{host}:{port}",
            "latency_ms": round((time.time() - start) * 1000),
        }
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"status": "down", "endpoint": f"{host}:{port}", "error": str(exc)}


def collect_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "robin_dir": str(ROBIN_DIR),
        "investigations_dir": str(INVESTIGATIONS_DIR),
        "tor_proxy": check_tor_proxy(),
        "presets": [],
        "models": [],
        "errors": [],
    }

    try:
        llm, llm_utils, _, _ = _load_pipeline_modules()
    except Exception as exc:  # noqa: BLE001 - missing deps are the common case
        status["errors"].append(
            f"could not import robin modules: {type(exc).__name__}: {exc}"
        )
        return status

    try:
        status["presets"] = list(getattr(llm, "PRESET_PROMPTS", {}).keys())
    except Exception as exc:  # noqa: BLE001
        status["errors"].append(f"presets unavailable: {exc}")

    try:
        status["models"] = list(llm_utils.get_model_choices())
        if not status["models"]:
            status["errors"].append(
                "no LLM models available — set an API key in the robin .env"
            )
    except Exception as exc:  # noqa: BLE001
        status["errors"].append(f"models unavailable: {exc}")

    return status


# --------------------------------------------------------------------------- #
# MCP surface
# --------------------------------------------------------------------------- #

TOOLS: list[types.Tool] = [
    types.Tool(
        name="robin_investigate",
        title="Robin dark-web investigation",
        description=(
            "Run the full Robin OSINT pipeline for one query: refine the query, "
            "search dark-web engines over Tor, LLM-filter the hits, scrape the "
            "top sources, and generate a preset-driven investigation summary. "
            "Returns the summary and the referenced sources. Requires Tor on "
            "127.0.0.1:9050 and a configured LLM API key."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Initial investigation query, e.g. 'acme corp breach'.",
                },
                "preset": {
                    "type": "string",
                    "description": (
                        "Investigation preset (default: threat_intel). Call "
                        "robin_status for the valid list."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "LLM model to use (default: first available). Call "
                        "robin_status for the valid list."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cap the number of search results before LLM filtering.",
                },
                "save": {
                    "type": "boolean",
                    "description": "Persist the investigation JSON to disk (default: true).",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="robin_status",
        title="Robin environment status",
        description=(
            "Report Robin's environment: available LLM models, valid presets, "
            "Tor SOCKS proxy reachability, and output directories. Use before "
            "robin_investigate to pick a valid preset/model or to diagnose "
            "failures."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
]


# Ordered pipeline stages, used for progress reporting (progress/total).
PIPELINE_STAGES = [
    "resolving model",
    "refining query",
    "searching dark-web engines",
    "filtering results with LLM",
    "scraping sources",
    "generating summary",
]


async def _notify(ctx: Any, message: str, step: int | None = None) -> None:
    """Best-effort progress notification; never fatal.

    ``step`` is a 1-based index into PIPELINE_STAGES, so progress runs 1..N of
    N. Uses the spec's progress notification (the logging capability is
    deprecated as of 2026-07-28) and is a no-op when the caller did not request
    progress.
    """
    _log(message)
    try:
        await ctx.session.report_progress(
            progress=float(step or 1),
            total=float(len(PIPELINE_STAGES)),
            message=message,
        )
    except Exception:  # noqa: BLE001 - client may not support progress
        pass


def _text_result(text: str, structured: dict[str, Any] | None = None, is_error: bool = False):
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=structured,
        is_error=is_error,
    )


async def list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    name = params.name
    args = params.arguments or {}

    if name == "robin_status":
        try:
            status = await asyncio.to_thread(collect_status)
        except BaseException as exc:  # noqa: BLE001 - never let SystemExit escape
            _log(f"robin_status failed: {exc}")
            return _text_result(f"robin_status failed: {exc}", is_error=True)
        return _text_result(json.dumps(status, indent=2), structured=status)

    if name == "robin_investigate":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _text_result(
                "robin_investigate requires a non-empty 'query' string.",
                is_error=True,
            )
        preset = args.get("preset") or "threat_intel"
        model = args.get("model")
        max_results = args.get("max_results")
        if max_results is not None:
            try:
                max_results = int(max_results)
                if max_results < 1:
                    raise ValueError
            except (TypeError, ValueError):
                return _text_result(
                    "'max_results' must be a positive integer.", is_error=True
                )
        save = bool(args.get("save", True))

        loop = asyncio.get_running_loop()

        def progress(msg: str, step: int | None = None) -> None:
            """Called from the worker thread; hop back to the loop to notify."""
            _log(msg)
            try:
                asyncio.run_coroutine_threadsafe(_notify(ctx, msg, step), loop)
            except RuntimeError:
                pass

        try:
            result = await asyncio.to_thread(
                run_pipeline,
                query,
                preset,
                model,
                max_results,
                save,
                progress,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced, never fatal
            _log(f"pipeline failed: {exc}\n{traceback.format_exc()}")
            return _text_result(
                f"Robin pipeline failed: {type(exc).__name__}: {exc}",
                structured={"error": str(exc), "query": query, "preset": preset},
                is_error=True,
            )
        return _text_result(render_result(result), structured=result)

    return _text_result(f"Unknown tool: {name}", is_error=True)


def build_server() -> Server:
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "Robin is a dark-web OSINT pipeline. Use robin_investigate with a "
            "short (<=5 word) query to run a full investigation; it can take "
            "several minutes because it searches .onion engines over Tor and "
            "scrapes the filtered sources. Call robin_status first if you need "
            "valid preset/model values or a Tor/credentials diagnosis."
        ),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def main() -> None:
    server = build_server()
    _log(f"serving stdio (robin_dir={ROBIN_DIR})")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
