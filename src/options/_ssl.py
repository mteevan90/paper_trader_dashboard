"""SSL trust-store injection helper for HTTPS calls on machines with TLS inspection.

Sibling copy of ``src/crypto/_ssl.py`` — kept parallel rather than shared so
each asset class stays self-contained. Consolidate to a single ``src/_ssl.py``
when a third asset class needs the same helper.
"""


def use_system_trust_store() -> None:
    """Inject truststore into Python's SSL module so HTTPS calls trust
    the OS certificate store rather than certifi's bundled CAs.

    Required on contributor machines where antivirus or corporate
    proxies do TLS inspection (re-sign certs with a CA that certifi's
    bundled list doesn't include — Norton 360 on Chris's machine is
    the canonical case). The OS trust store is always a valid
    superset of certifi's bundle, so calling this on machines without
    TLS inspection is a no-op in practice.

    Idempotent — safe to call multiple times. Must be invoked before
    any imports that establish SSL connections (requests sessions,
    HTTP clients, etc.). Library modules do not call this on import;
    entry-point scripts call it at the top.
    """
    import truststore
    truststore.inject_into_ssl()
