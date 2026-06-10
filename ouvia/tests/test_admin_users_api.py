from __future__ import annotations


def test_admin_required_for_users_pages(client, make_user, login):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    # Not logged in -> redirect to login
    resp = client.get("/users", follow_redirects=False)
    assert resp.status_code in (302, 303)

    # Logged in as non-admin -> 403
    login("sender1")
    resp2 = client.get("/users")
    assert resp2.status_code == 403

    # Admin can access
    login("admin1", "secret")
    ok = client.get("/users")
    assert ok.status_code == 200


def test_admin_create_user_json_validation(client, make_user, login):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    login("admin1", "secret")

    # Missing unique_id
    resp = client.post("/users/create", json={"role": "sender"})
    assert resp.status_code == 400

    # Invalid role
    resp2 = client.post("/users/create", json={"unique_id": "x", "role": "nope"})
    assert resp2.status_code == 400

    # Admin must have password
    resp3 = client.post("/users/create", json={"unique_id": "admin2", "role": "admin"})
    assert resp3.status_code == 400

    # Sender ok
    ok = client.post("/users/create", json={"unique_id": "s2", "role": "sender"})
    assert ok.status_code == 200
    payload = ok.get_json()
    assert payload["success"] is True
    assert payload["user"]["unique_id"] == "s2"


def test_admin_update_user_self_demote_blocked(client, make_user, login):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    login("admin1", "secret")

    from ouvia.models import User

    admin = User.query.filter_by(unique_id="admin1").first()
    assert admin is not None

    resp = client.post(
        f"/users/{admin.id}/update",
        json={"unique_id": "admin1", "role": "sender"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_admin_delete_user_rules(client, make_user, login):
    from ouvia.models import User

    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    make_user(unique_id="admin2", role="admin", password="secret2", accept_terms=True)
    make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    login("admin1", "secret")

    admin1 = User.query.filter_by(unique_id="admin1").first()
    admin2 = User.query.filter_by(unique_id="admin2").first()
    sender = User.query.filter_by(unique_id="sender1").first()
    assert admin1 and admin2 and sender

    # Cannot delete self
    resp = client.post(f"/users/{admin1.id}/delete")
    assert resp.status_code == 400

    # Can delete non-admin user
    ok = client.post(f"/users/{sender.id}/delete")
    assert ok.status_code == 200
    assert ok.get_json()["success"] is True

    # Can delete one admin if another remains
    ok2 = client.post(f"/users/{admin2.id}/delete")
    assert ok2.status_code == 200
    assert ok2.get_json()["success"] is True

    # Remaining admin can't delete itself (still blocked)
    resp2 = client.post(f"/users/{admin1.id}/delete")
    assert resp2.status_code == 400


def test_user_deletion_info_receiver_and_validator(
    client, db, make_user, login, seed_conversation
):
    from ouvia.constants import (
        ENTRY_STATUS_EVALUATED,
        ENTRY_STATUS_TRANSLATED,
        ENTRY_STATUS_VALIDATED,
    )
    from ouvia.models import (
        ConversationEntry,
        ReceiverResponse,
        User,
        ValidatorResponseEvaluation,
        ValidatorReview,
    )

    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    login("admin1", "secret")

    conv, questions = seed_conversation(external_id="C1", question_count=2)
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )
    receiver = make_user(unique_id="r1", role="receiver", accept_terms=True)
    validator = make_user(unique_id="v1", role="validator", accept_terms=True)

    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        receiver_id=receiver.id,
        validator_id=validator.id,
        status=ENTRY_STATUS_EVALUATED,
        translation_text="t",
    )
    db.session.add(entry)
    db.session.commit()

    # Add receiver responses
    for q in questions:
        db.session.add(
            ReceiverResponse(
                conversation_entry_id=entry.id,
                question_id=q.id,
                response_text="x",
            )
        )
    db.session.commit()

    # Receiver deletion-info
    info = client.get(f"/users/{receiver.id}/deletion-info")
    assert info.status_code == 200
    payload = info.get_json()
    assert payload["role"] == "receiver"
    assert payload["total_entries"] == 1
    assert payload["total_responses"] == 2
    assert payload["entries_to_rollback"] == 1

    # Create validator artifacts and move entry to validated
    entry.status = ENTRY_STATUS_VALIDATED
    db.session.add(
        ValidatorReview(
            conversation_entry_id=entry.id,
            validator_id=validator.id,
            translation_score=80,
            audio_text_mismatch=False,
        )
    )
    for q in questions:
        db.session.add(
            ValidatorResponseEvaluation(
                conversation_entry_id=entry.id,
                question_id=q.id,
                is_correct=True,
            )
        )
    db.session.commit()

    info2 = client.get(f"/users/{validator.id}/deletion-info")
    assert info2.status_code == 200
    payload2 = info2.get_json()
    assert payload2["role"] == "validator"
    assert payload2["total_entries"] == 1
    assert payload2["total_evaluations"] == 2
    assert payload2["entries_to_rollback"] == 1

    # Delete receiver: should remove responses and roll back
    del_receiver = client.post(f"/users/{receiver.id}/delete")
    assert del_receiver.status_code == 200

    entry2 = db.session.get(ConversationEntry, entry.id)
    assert entry2 is not None
    assert entry2.receiver_id is None
    # Was validated, receiver deletion only rolls back evaluated->translated; keep it stable by re-setting.
    # Set to evaluated then delete receiver was above; receiver deletion already ran when status was validated.
    # So for receiver deletion behavior, create a fresh entry in translated/evaluated state.

    # Fresh entry for receiver rollback check (use a different conversation to
    # avoid the unique constraint on (conversation_id, sender_id)).
    receiver2 = make_user(unique_id="r2", role="receiver", accept_terms=True)
    conv2, _ = seed_conversation(external_id="C2", question_count=2)
    entry3 = ConversationEntry(
        conversation_id=conv2.id,
        sender_id=sender.id,
        receiver_id=receiver2.id,
        status=ENTRY_STATUS_EVALUATED,
        translation_text="t",
    )
    db.session.add(entry3)
    db.session.commit()

    db.session.add(
        ReceiverResponse(
            conversation_entry_id=entry3.id,
            question_id=questions[0].id,
            response_text="a",
        )
    )
    db.session.add(
        ReceiverResponse(
            conversation_entry_id=entry3.id,
            question_id=questions[1].id,
            response_text="b",
        )
    )
    db.session.commit()

    del_receiver2 = client.post(f"/users/{receiver2.id}/delete")
    assert del_receiver2.status_code == 200

    entry3r = db.session.get(ConversationEntry, entry3.id)
    assert entry3r is not None
    assert entry3r.receiver_id is None
    assert entry3r.status == ENTRY_STATUS_TRANSLATED
    assert (
        ReceiverResponse.query.filter_by(conversation_entry_id=entry3.id).count() == 0
    )

    # Delete validator: should delete evaluations/review and roll back validated->evaluated
    del_validator = client.post(f"/users/{validator.id}/delete")
    assert del_validator.status_code == 200

    entry4 = db.session.get(ConversationEntry, entry.id)
    assert entry4 is not None
    assert entry4.validator_id is None
    assert entry4.status == ENTRY_STATUS_EVALUATED
    assert ValidatorReview.query.filter_by(conversation_entry_id=entry.id).count() == 0
    assert (
        ValidatorResponseEvaluation.query.filter_by(
            conversation_entry_id=entry.id
        ).count()
        == 0
    )
