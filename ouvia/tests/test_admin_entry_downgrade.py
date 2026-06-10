from __future__ import annotations

from pathlib import Path


def test_admin_can_downgrade_entry_and_cleanup_artifacts(
    app, client, db, make_user, login, seed_conversation
):
    from ouvia.constants import (
        ENTRY_STATUS_COMPLETED,
        ENTRY_STATUS_EVALUATED,
        ENTRY_STATUS_READY,
        ENTRY_STATUS_TRANSLATED,
    )
    from ouvia.models import (
        ConversationEntry,
        ReceiverResponse,
        Recording,
        SenderFinalSurvey,
        ValidatorResponseEvaluation,
        ValidatorReview,
    )

    admin = make_user(
        unique_id="admin1", role="admin", password="secret", accept_terms=True
    )
    sender = make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )
    receiver = make_user(unique_id="receiver1", role="receiver", accept_terms=True)
    validator = make_user(unique_id="validator1", role="validator", accept_terms=True)

    conv, questions = seed_conversation(external_id="C1", question_count=2)
    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        receiver_id=receiver.id,
        validator_id=validator.id,
        status=ENTRY_STATUS_COMPLETED,
        translation_text="translated text",
        translation_model="model-x",
    )
    db.session.add(entry)
    db.session.flush()

    # Create recording on disk + DB
    recordings_dir = Path(app.config["RECORDINGS_DIR"])
    entry_dir = recordings_dir / f"entry_{entry.id}"
    entry_dir.mkdir(parents=True, exist_ok=True)
    stored_rel = Path(f"entry_{entry.id}") / "entry_test.wav"
    file_path = recordings_dir / stored_rel
    file_path.write_bytes(b"RIFF" + b"\x00" * 32)
    db.session.add(
        Recording(
            conversation_entry_id=entry.id,
            stored_filename=str(stored_rel),
            original_filename="entry_test.wav",
            file_mimetype="audio/wav",
            file_size=file_path.stat().st_size,
        )
    )

    # Receiver responses (evaluated+)
    for q in questions:
        db.session.add(
            ReceiverResponse(
                conversation_entry_id=entry.id,
                question_id=q.id,
                response_text="resp",
            )
        )

    # Validator review + per-question evaluations (validated+)
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

    # Sender final survey (completed)
    db.session.add(
        SenderFinalSurvey(
            conversation_entry_id=entry.id,
            sender_id=sender.id,
            baseline_satisfaction_score=3,
            baseline_trust_score=3,
            baseline_reliance_score=3,
            satisfaction_score=4,
            trust_score=4,
            reliance_score=4,
        )
    )

    db.session.commit()
    assert file_path.exists()

    login(admin.unique_id, "secret")

    # Downgrade completed -> evaluated: removes final survey + validator data, keeps receiver responses/translation/recording.
    resp = client.post(
        f"/users/{sender.id}/entries/{entry.id}/downgrade",
        data={"target_status": ENTRY_STATUS_EVALUATED},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)

    refreshed = db.session.get(ConversationEntry, entry.id)
    assert refreshed is not None
    assert refreshed.status == ENTRY_STATUS_EVALUATED
    assert refreshed.translation_text == "translated text"
    assert refreshed.translation_model == "model-x"
    assert refreshed.recording is not None
    assert file_path.exists()

    assert (
        SenderFinalSurvey.query.filter_by(conversation_entry_id=entry.id).count() == 0
    )
    assert ValidatorReview.query.filter_by(conversation_entry_id=entry.id).count() == 0
    assert (
        ValidatorResponseEvaluation.query.filter_by(
            conversation_entry_id=entry.id
        ).count()
        == 0
    )
    assert ReceiverResponse.query.filter_by(
        conversation_entry_id=entry.id
    ).count() == len(questions)

    # Downgrade evaluated -> ready: removes receiver responses + translation + recording (DB + disk).
    resp2 = client.post(
        f"/users/{sender.id}/entries/{entry.id}/downgrade",
        data={"target_status": ENTRY_STATUS_READY},
        follow_redirects=False,
    )
    assert resp2.status_code in (302, 303)

    refreshed2 = db.session.get(ConversationEntry, entry.id)
    assert refreshed2 is not None
    assert refreshed2.status == ENTRY_STATUS_READY
    assert refreshed2.translation_text is None
    assert refreshed2.translation_model is None
    assert refreshed2.recording is None
    assert not file_path.exists()

    assert ReceiverResponse.query.filter_by(conversation_entry_id=entry.id).count() == 0


def test_ready_entries_cannot_be_downgraded(
    client, db, make_user, login, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_READY
    from ouvia.models import ConversationEntry

    admin = make_user(
        unique_id="admin1", role="admin", password="secret", accept_terms=True
    )
    sender = make_user(
        unique_id="sender1", role="sender", gender="man", trans="yes", accept_terms=True
    )
    conv, _ = seed_conversation(external_id="C1")
    entry = ConversationEntry(
        conversation_id=conv.id, sender_id=sender.id, status=ENTRY_STATUS_READY
    )
    db.session.add(entry)
    db.session.commit()

    login(admin.unique_id, "secret")
    resp = client.post(
        f"/users/{sender.id}/entries/{entry.id}/downgrade",
        data={"target_status": ENTRY_STATUS_READY},
        follow_redirects=False,
    )
    assert resp.status_code == 400
