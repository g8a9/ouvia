import pytest


def test_create_app_imports_and_has_routes():
    """Smoke test: importing the package and creating the Flask app should succeed
    and register at least one expected admin route (dashboard).
    """
    try:
        from ouvia import create_app
    except Exception as exc:  # allow test to fail with clear message
        pytest.fail(f"Failed to import create_app from package 'ouvia': {exc}")

    app = create_app()
    assert app is not None
    assert hasattr(app, "url_map")

    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    # Validate a few key endpoints exist
    assert "dashboard" in endpoints or "ouvia.dashboard" in endpoints
