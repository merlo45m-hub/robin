#!/usr/bin/env python3
"""Search-result relevance: the parser must not return navigation as results.

Background: a nonsense query and a real query returned identical results, because the
parser collected every .onion anchor on the engine page - including the marketplace
category menu that appears on all of them - with no relevance check at all.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import search  # noqa: E402


class RelevanceTermsTest(unittest.TestCase):

    def test_stopwords_are_dropped(self):
        self.assertEqual(search._relevance_terms("how to unlock the samsung"), ["unlock", "samsung"])

    def test_short_and_uppercase_tokens_normalised(self):
        self.assertEqual(search._relevance_terms("S23 ULTRA"), ["s23", "ultra"])

    def test_blank_query_yields_no_terms(self):
        self.assertEqual(search._relevance_terms(""), [])
        self.assertEqual(search._relevance_terms(None), [])


class RelevanceScoreTest(unittest.TestCase):

    def test_title_match_outranks_url_match(self):
        terms = ["samsung"]
        self.assertEqual(search._relevance_score("Samsung S23 unlock", "http://a.onion", terms), 2)
        self.assertEqual(search._relevance_score("Unrelated", "http://samsung.onion", terms), 1)

    def test_no_match_scores_zero(self):
        self.assertEqual(
            search._relevance_score("Marketplaces", "http://example.onion", ["samsung"]),
            0,
        )

    def test_no_terms_scores_neutral(self):
        self.assertEqual(search._relevance_score("anything", "http://x.onion", []), 1)


class NavigationTitleTest(unittest.TestCase):

    def test_category_labels_are_navigation(self):
        for label in ("Marketplaces", "Directories", "Hacking", "Categories", "Home"):
            self.assertTrue(search._is_navigation_title(label), label)

    def test_real_titles_are_not_navigation(self):
        for title in (
            "Samsung S23 unlock service",
            "THE HIGH TABLE - Hacking services",
            "Arcadia Marketplace",
        ):
            self.assertFalse(search._is_navigation_title(title), title)


class FetchRelevanceTest(unittest.TestCase):

    def _response(self, html: str):
        resp = mock.Mock()
        resp.status_code = 200
        resp.text = html
        return resp

    def _engine(self) -> str:
        return "http://engine.onion/search?q={query}"

    def test_navigation_links_are_excluded(self):
        html = """
        <html><body>
          <a href="http://amndir7jfxnt5glt.onion/?cat=1">Marketplaces</a>
          <a href="http://real.onion/unlock">Samsung S23 unlock service</a>
        </body></html>
        """
        session = mock.Mock()
        session.get.return_value = self._response(html)
        with mock.patch.object(search, "get_tor_session", return_value=session):
            results = search.fetch_search_results(self._engine(), "samsung s23 unlock")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["link"], "http://real.onion/unlock")

    def test_irrelevant_links_are_ranked_below_relevant_ones(self):
        html = """
        <html><body>
          <a href="http://weak.onion/page">samsung mention in url only</a>
          <a href="http://strong.onion/x">Samsung S23 unlock guide</a>
        </body></html>
        """
        session = mock.Mock()
        session.get.return_value = self._response(html)
        with mock.patch.object(search, "get_tor_session", return_value=session):
            results = search.fetch_search_results(self._engine(), "samsung unlock")
        self.assertEqual(results[0]["link"], "http://strong.onion/x")

    def test_no_relevant_hits_returns_empty_rather_than_navigation(self):
        html = """
        <html><body>
          <a href="http://other.onion/a">Something else entirely</a>
        </body></html>
        """
        session = mock.Mock()
        session.get.return_value = self._response(html)
        with mock.patch.object(search, "get_tor_session", return_value=session):
            results = search.fetch_search_results(self._engine(), "samsung unlock")
        # Returning the unfiltered set here is what produced the original symptom: a
        # nonsense query and a real one both returned the site's category menu.
        self.assertEqual(results, [])

    def test_nonsense_query_does_not_return_the_same_navigation_results(self):
        # The regression that prompted the relevance filter: a nonsense query returned
        # the marketplace category menu, identical to a real query's results.
        nav_only = """
        <html><body>
          <a href="http://amndir7jfxnt5glt.onion/?cat=1">Marketplaces</a>
          <a href="http://amndir7jfxnt5glt.onion/?cat=2">Directories</a>
          <a href="http://amndir7jfxnt5glt.onion/?cat=3">Hacking</a>
        </body></html>
        """
        session = mock.Mock()
        session.get.return_value = self._response(nav_only)
        with mock.patch.object(search, "get_tor_session", return_value=session):
            results = search.fetch_search_results(self._engine(), "zzzqqqxxx nonexistent")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
