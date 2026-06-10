from __future__ import annotations

import pytest


def test_index_redirects_to_login_when_logged_out(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert "/login" in resp.headers.get("Location", "")


def test_login_rejects_unknown_user(client):
    resp = client.post("/login", data={"unique_id": "missing"}, follow_redirects=True)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Invalid credentials" in body


def test_login_requires_password_for_admin(client, make_user):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)

    bad = client.post(
        "/login",
        data={"unique_id": "admin1", "password": "wrong"},
        follow_redirects=True,
    )
    assert bad.status_code == 200
    assert "Invalid credentials" in bad.get_data(as_text=True)

    ok = client.post(
        "/login",
        data={"unique_id": "admin1", "password": "secret"},
        follow_redirects=False,
    )
    assert ok.status_code in (302, 303)


@pytest.mark.parametrize(
    "role,authkey",
    [
        ("sender", "sender-key"),
        ("receiver", "receiver-key"),
        ("validator", "validator-key"),
    ],
)
def test_signup_requires_terms_and_prolific_id(client, role, authkey):
    # Missing terms
    resp = client.post(
        f"/signup?role={role}&authkey={authkey}",
        data={"prolific_id": "P1"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    # Missing prolific id
    resp2 = client.post(
        f"/signup?role={role}&authkey={authkey}",
        data={"accept_terms": "on"},
        follow_redirects=True,
    )
    assert resp2.status_code == 200


def test_signup_rejects_wrong_role_or_key(client):
    assert client.get("/signup?role=unknown&authkey=x").status_code == 404
    assert client.get("/signup?role=sender&authkey=wrong").status_code == 403


def test_sender_signup_creates_user_and_assigns_entry(client, db, seed_conversation):
    from ouvia.models import ConversationEntry, User

    # Sender signup assigns DEFAULT_SENDER_ASSIGNMENTS conversations (default=10).
    for i in range(10):
        seed_conversation(external_id=f"C{i + 1}", question_count=1)

    resp = client.post(
        "/signup?role=sender&authkey=sender-key",
        data={
            "accept_terms": "on",
            "prolific_id": "sender1",
            "gender": "man",
            "trans": "yes",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)

    user = User.query.filter_by(unique_id="sender1").first()
    assert user is not None
    assert user.role == "sender"
    assert user.terms_accepted_at is not None

    entry = ConversationEntry.query.filter_by(sender_id=user.id).first()
    assert entry is not None


def test_receiver_signup_assigns_translated_entry(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_TRANSLATED
    from ouvia.models import ConversationEntry, User

    # Receiver signup assigns DEFAULT_RECEIVER_ASSIGNMENTS entries (default=3).
    convs = [
        seed_conversation(external_id=f"C{i + 1}", question_count=1)[0]
        for i in range(3)
    ]
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    for conv in convs:
        db.session.add(
            ConversationEntry(
                conversation_id=conv.id,
                sender_id=sender.id,
                status=ENTRY_STATUS_TRANSLATED,
                translation_text="t",
            )
        )
    db.session.commit()

    resp = client.post(
        "/signup?role=receiver&authkey=receiver-key",
        data={"accept_terms": "on", "prolific_id": "r1"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)

    receiver = User.query.filter_by(unique_id="r1").first()
    assert receiver is not None
    assigned = ConversationEntry.query.filter_by(receiver_id=receiver.id).all()
    assert len(assigned) == 3


def test_terms_enforcement_redirects_until_accepted(client, db, make_user):
    from ouvia.models import AppSetting

    # Enable terms enforcement
    db.session.add(AppSetting(key="require_terms_acceptance", value="1"))
    db.session.commit()

    user = make_user(unique_id="u1", role="receiver", accept_terms=False)

    # Login without password (non-admin) is allowed
    login_resp = client.post(
        "/login", data={"unique_id": user.unique_id}, follow_redirects=False
    )
    assert login_resp.status_code in (302, 303)

    # Access to /home should redirect to /terms
    resp = client.get("/home", follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert resp.headers.get("Location", "").endswith("/terms")

    # Accept terms
    accept = client.post("/terms", follow_redirects=False)
    assert accept.status_code in (302, 303)

    # Now /home works
    ok = client.get("/home")
    assert ok.status_code == 200
