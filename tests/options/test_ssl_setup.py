"""Tests for src/options/_ssl.py truststore injection.

The injection mutates module-level state in Python's ``ssl`` module.
Because pytest runs in a single process, the injection from one test
can leak into others. We exercise behavior here rather than restoring
state — once injected, ``ssl.create_default_context`` produces a
truststore context for the remainder of the test session, which
matches the expected runtime behavior of an entry-point script.
"""

from __future__ import annotations

import ssl

import truststore

from src.options._ssl import use_system_trust_store


def test_use_system_trust_store_runs_without_exception():
    use_system_trust_store()


def test_use_system_trust_store_is_idempotent():
    """truststore documents inject_into_ssl as safe to call repeatedly."""
    use_system_trust_store()
    use_system_trust_store()
    use_system_trust_store()


def test_default_context_is_truststore_after_inject():
    """After injection, ssl.create_default_context returns a context
    whose class is truststore.SSLContext — i.e. cert verification will
    consult the OS trust store instead of certifi's bundle."""
    use_system_trust_store()
    ctx = ssl.create_default_context()
    assert isinstance(ctx, truststore.SSLContext)
