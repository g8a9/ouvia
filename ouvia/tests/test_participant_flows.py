from __future__ import annotations

from io import BytesIO

import pytest


def test_participant_home_redirects_admin_to_dashboard(client, make_user):
    make_user(unique_id="admin1", role="admin", password="secret", accept_terms=True)
    client.post("/login", data={"unique_id": "admin1", "password": "secret"})

    resp = client.get("/home", follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert resp.headers.get("Location", "").endswith("/dashboard")


def test_sender_recording_upload_creates_record_and_updates_status(
    client, db, make_user, seed_conversation, make_audio_file
):
    from ouvia.constants import ENTRY_STATUS_READY, ENTRY_STATUS_RECORDED
    from ouvia.models import ConversationEntry, Recording

    conv, _ = seed_conversation(external_id="C1")
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    entry = ConversationEntry(
        conversation_id=conv.id, sender_id=sender.id, status=ENTRY_STATUS_READY
    )
    db.session.add(entry)
    db.session.commit()

    client.post("/login", data={"unique_id": "s1"}, follow_redirects=False)

    bio, fname = make_audio_file()
    data = {"audio_data": (bio, fname)}

    resp = client.post(
        f"/entries/{entry.id}/recordings", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    entry2 = db.session.get(ConversationEntry, entry.id)
    assert entry2 is not None
    assert entry2.status == ENTRY_STATUS_RECORDED

    rec = Recording.query.filter_by(conversation_entry_id=entry.id).first()
    assert rec is not None
    assert rec.file_size and rec.file_size > 0


def test_receiver_save_responses_sets_evaluated_status_and_next_url(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_EVALUATED, ENTRY_STATUS_TRANSLATED
    from ouvia.models import ConversationEntry, ReceiverResponse

    conv, questions = seed_conversation(external_id="C1", question_count=2)
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )
    receiver = make_user(unique_id="r1", role="receiver", accept_terms=True)

    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        receiver_id=receiver.id,
        status=ENTRY_STATUS_TRANSLATED,
        translation_text="translation",
    )
    db.session.add(entry)
    db.session.commit()

    client.post("/login", data={"unique_id": "r1"}, follow_redirects=False)

    resp = client.post(
        f"/entries/{entry.id}/receiver_responses",
        json={
            "responses": {
                str(questions[0].id): "a1",
                str(questions[1].id): "a2",
            }
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["complete"] is True

    entry2 = db.session.get(ConversationEntry, entry.id)
    assert entry2 is not None
    assert entry2.status == ENTRY_STATUS_EVALUATED

    assert ReceiverResponse.query.filter_by(conversation_entry_id=entry.id).count() == 2


def test_receiver_save_responses_rejects_missing_answers(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_TRANSLATED
    from ouvia.models import ConversationEntry

    conv, questions = seed_conversation(external_id="C1", question_count=2)
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )
    receiver = make_user(unique_id="r1", role="receiver", accept_terms=True)

    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        receiver_id=receiver.id,
        status=ENTRY_STATUS_TRANSLATED,
        translation_text="translation",
    )
    db.session.add(entry)
    db.session.commit()

    client.post("/login", data={"unique_id": "r1"}, follow_redirects=False)

    resp = client.post(
        f"/entries/{entry.id}/receiver_responses",
        json={"responses": {str(questions[0].id): "a1", str(questions[1].id): ""}},
    )
    assert resp.status_code == 400


def test_validator_review_creates_review_evaluations_and_sets_validated(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_EVALUATED, ENTRY_STATUS_VALIDATED
    from ouvia.models import (
        ConversationEntry,
        ValidatorResponseEvaluation,
        ValidatorReview,
    )

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

    client.post("/login", data={"unique_id": "v1"}, follow_redirects=False)

    resp = client.post(
        f"/entries/{entry.id}/validator_review",
        json={
            "translation_score": 80,
            "audio_text_mismatch": False,
            "answers": {str(questions[0].id): True, str(questions[1].id): False},
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    entry2 = db.session.get(ConversationEntry, entry.id)
    assert entry2 is not None
    assert entry2.status == ENTRY_STATUS_VALIDATED

    assert ValidatorReview.query.filter_by(conversation_entry_id=entry.id).count() == 1
    assert (
        ValidatorResponseEvaluation.query.filter_by(
            conversation_entry_id=entry.id
        ).count()
        == 2
    )


def test_sender_final_survey_completes_entry_and_assigns_confirmation_code(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_COMPLETED, ENTRY_STATUS_VALIDATED
    from ouvia.models import ConversationEntry, SenderFinalSurvey, User

    conv, questions = seed_conversation(external_id="C1", question_count=2)
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        final_sender_id=sender.id,
        status=ENTRY_STATUS_VALIDATED,
        translation_text="t",
    )
    db.session.add(entry)
    db.session.commit()

    client.post("/login", data={"unique_id": "s1"}, follow_redirects=False)

    # Baseline step
    b = client.post(
        f"/entries/{entry.id}/final-survey",
        data={
            "survey_step": "baseline",
            "baseline_satisfaction_score": "3",
            "baseline_trust_score": "3",
            "baseline_reliance_score": "4",
        },
        follow_redirects=False,
    )
    assert b.status_code in (302, 303)

    survey = SenderFinalSurvey.query.filter_by(conversation_entry_id=entry.id).first()
    assert survey is not None
    assert survey.baseline_trust_score == 3
    assert survey.baseline_reliance_score == 4
    assert survey.baseline_satisfaction_score == 3

    # Final step completes the entry and assigns confirmation code
    f = client.post(
        f"/entries/{entry.id}/final-survey",
        data={
            "survey_step": "final",
            "satisfaction_score": "4",
            "trust_score": "5",
            "reliance_score": "2",
        },
        follow_redirects=False,
    )
    assert f.status_code in (302, 303)

    survey2 = SenderFinalSurvey.query.filter_by(conversation_entry_id=entry.id).first()
    assert survey2 is not None
    assert survey2.satisfaction_score == 4
    assert survey2.trust_score == 5
    assert survey2.reliance_score == 2

    entry2 = db.session.get(ConversationEntry, entry.id)
    assert entry2 is not None
    assert entry2.status == ENTRY_STATUS_COMPLETED

    sender2 = User.query.filter_by(unique_id="s1").first()
    assert sender2 is not None
    assert sender2.confirmation_code == "FSENDER001"


def test_sender_final_survey_rejects_invalid_scores(
    client, db, make_user, seed_conversation
):
    from ouvia.constants import ENTRY_STATUS_VALIDATED
    from ouvia.models import ConversationEntry

    conv, _ = seed_conversation(external_id="C1", question_count=1)
    sender = make_user(
        unique_id="s1", role="sender", gender="man", trans="yes", accept_terms=True
    )

    entry = ConversationEntry(
        conversation_id=conv.id,
        sender_id=sender.id,
        final_sender_id=sender.id,
        status=ENTRY_STATUS_VALIDATED,
        translation_text="t",
    )
    db.session.add(entry)
    db.session.commit()

    client.post("/login", data={"unique_id": "s1"}, follow_redirects=False)

    # Out of range should not redirect
    resp = client.post(
        f"/entries/{entry.id}/final-survey",
        data={
            "survey_step": "baseline",
            "baseline_satisfaction_score": "0",
            "baseline_trust_score": "4",
            "baseline_reliance_score": "3",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
