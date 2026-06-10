from __future__ import annotations


def test_dashboard_requires_admin(client, make_user, login, logout):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    # Not logged in
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code in (302, 303)

    # Non-admin
    login("sender1")
    assert client.get("/dashboard").status_code == 403

    # Admin
    login("admin1", "secret")
    ok = client.get("/dashboard")
    assert ok.status_code == 200


def test_admin_stats_page_renders(client, make_user, login):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    login("admin1", "secret")

    resp = client.get("/admin/stats")
    assert resp.status_code == 200


def test_export_conversation_entries_csv(client, db, make_user, seed_conversation):
    from ouvia.models import ConversationEntry

    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)

    conv, _ = seed_conversation(external_id="C1")
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    db.session.add(ConversationEntry(conversation_id=conv.id, sender_id=sender.id))
    db.session.commit()

    client.post(
        "/login",
        data={"unique_id": "admin1", "password": "secret"},
        follow_redirects=False,
    )

    resp = client.get("/conversation-entries/export")
    assert resp.status_code == 200
    assert resp.headers.get("Content-Type", "").startswith("text/csv")
    text = resp.get_data(as_text=True)
    assert "conversation_entry_id" in text.splitlines()[0]


def test_admin_preview_routes_require_admin(client, make_user, login):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    # Non-admin blocked
    login("sender1")
    assert client.get("/admin/preview").status_code == 403

    # Admin allowed
    login("admin1", "secret")
    assert client.get("/admin/preview").status_code == 200
    assert client.get("/admin/preview/receiver").status_code == 200
    assert client.get("/admin/preview/validator").status_code == 200
