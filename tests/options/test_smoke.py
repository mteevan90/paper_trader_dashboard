"""Phase 1 smoke test: assert that ``src.options`` imports cleanly."""


def test_options_package_imports():
    import src.options as options  # noqa: F401

    assert options is not None
