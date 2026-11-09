#!/usr/bin/env python3
"""Regression tests for the search-timeout budget.

Background: /investigate returned a deterministic 504 because the search stage alone
fanned out to ~11 onion engines at a 40s per-engine budget with 3 retries and only 5
workers (~84s measured), before any scraping or LLM calls. The Android client asked
for a 60s overall budget, so the pipeline could never finish.

These tests pin the budget so it cannot silently regress. They mock the network: no
Tor, no engine calls, no LLM.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import search  # noqa: E402


class EngineBudgetTest(unittest.TestCase):
    """The per-engine budget must stay bounded."""

    def test_engine_timeout_is_bounded(self):
        # A single slow engine must not be able to hold a worker for the better part
        # of a minute. 15s is the value the fix settled on.
        self.assertLessEqual(search.ENGINE_TIMEOUT_S, 20)

    def test_engine_retries_are_bounded(self):
        # Retries multiply the per-engine budget; more than a couple turns one dead
        # engine into a multi-minute stall.
        self.assertLessEqual(search.ENGINE_RETRIES, 2)

    def test_fetch_uses_the_bounded_timeout(self):
        """fetch_search_results must pass ENGINE_TIMEOUT_S, not a hardcoded value."""
        session = mock.Mock()
        response = mock.Mock()
        response.status_code = 200
        response.text = "<html><body>no links</body></html>"
        session.get.return_value = response

        with mock.patch.object(search, "get_tor_session", return_value=session):
            search.fetch_search_results("http://example.onion/search?q={query}", "test")

        _, kwargs = session.get.call_args
        self.assertEqual(kwargs.get("timeout"), search.ENGINE_TIMEOUT_S)

    def test_tor_session_is_cached_per_thread(self):
        """Each worker thread should reuse one session rather than rebuild it."""
        first = search.get_tor_session()
        second = search.get_tor_session()
        self.assertIs(first, second)

        # A different thread must get its own session (a shared requests.Session
        # across a worker pool is not safe).
        other: list = []
        t = threading.Thread(target=lambda: other.append(search.get_tor_session()))
        t.start()
        t.join(timeout=10)
        self.assertEqual(len(other), 1)
        self.assertIsNot(other[0], first)


class HangingEngineTest(unittest.TestCase):
    """A hung or failing engine must not stall the fan-out."""

    def test_timeout_returns_empty_not_raises(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.Timeout("engine hung")
        with mock.patch.object(search, "get_tor_session", return_value=session):
            result = search.fetch_search_results("http://example.onion/search?q={query}", "test")
        self.assertEqual(result, [])

    def test_connection_error_returns_empty(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.ConnectionError("no route")
        with mock.patch.object(search, "get_tor_session", return_value=session):
            result = search.fetch_search_results("http://example.onion/search?q={query}", "test")
        self.assertEqual(result, [])

    def test_fan_out_completes_when_engines_are_slow(self):
        """Whole search must stay well inside a 60s caller budget."""
        started = time.monotonic()

        def slow_fetch(endpoint, query):
            time.sleep(0.3)  # stand-in for a slow onion engine
            return []

        with mock.patch.object(search, "fetch_search_results", side_effect=slow_fetch):
            results = search.get_search_results("test query")

        elapsed = time.monotonic() - started
        self.assertEqual(results, [])
        # 20 engines at 0.3s with 8 workers is ~1s; anything near 60s means the
        # parallelism regressed.
        self.assertLess(elapsed, 20.0, f"search fan-out took {elapsed:.1f}s")

    def test_more_workers_than_before(self):
        """Parallelism must be high enough to cover the engine list."""
        import inspect

        sig = inspect.signature(search.get_search_results)
        self.assertGreaterEqual(sig.parameters["max_workers"].default, 8)


class ApiTimeoutGuardTest(unittest.TestCase):
    """The /search endpoint must not be able to hang a worker indefinitely."""

    def test_search_endpoint_has_a_timeout_guard(self):
        api_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robin_api.py"
        )
        if not os.path.exists(api_path):
            self.skipTest("robin_api.py not present")
        src = open(api_path, encoding="utf-8").read()

        # Locate the /search endpoint body and confirm it is wrapped.
        start = src.index("async def search_endpoint")
        end = src.index("async def scrape_endpoint")
        body = src[start:end]
        self.assertIn("asyncio.wait_for", body, "/search must be bounded by a timeout")
        self.assertIn("asyncio.TimeoutError", body, "/search must handle the timeout")

    def test_search_request_has_a_timeout_field(self):
        api_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robin_api.py"
        )
        if not os.path.exists(api_path):
            self.skipTest("robin_api.py not present")
        src = open(api_path, encoding="utf-8").read()
        start = src.index("class SearchRequest")
        end = src.index("class ScrapeRequest")
        self.assertIn("timeout", src[start:end])


if __name__ == "__main__":
    unittest.main(verbosity=2)
