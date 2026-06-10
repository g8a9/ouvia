from __future__ import annotations

from io import BytesIO
from typing import Callable

import pytest
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def app(tmp_path):
    """Flask app configured for tests using in-memory SQLite.

    Important: we use StaticPool so the same in-memory DB connection is reused
    across requests; otherwise SQLite would present a fresh empty DB per request.
    """

    # Reset module-level caches/flags that may persist across tests due to the
    # global Flask app pattern in this codebase.
    import ouvia.app as app_module
    from ouvia import create_app

    app_module._SETTINGS_TABLE_READY = False
    try:
        app_module._RECEIVER_LOCALE_CACHE.clear()
    except Exception:
        pass

    overrides = {
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "SQLALCHEMY_ENGINE_OPTIONS": {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        },
        "RECORDINGS_DIR": str(tmp_path / "recordings"),
        "TRANSLATIONS_DIR": str(tmp_path / "translations"),
        "DEFAULT_SENDER_ASSIGNMENTS": 1,
        "DEFAULT_RECEIVER_ASSIGNMENTS": 1,
        "DEFAULT_VALIDATOR_ASSIGNMENTS": 1,
        "ROLE_SIGNUP_KEYS": {
            "sender": "sender-key",
            "final_sender": "final-sender-key",
            "receiver": "receiver-key",
            "validator": "validator-key",
        },
        "ROLE_CONFIRMATION_CODES": {
            "sender": "SENDER001",
            "final_sender": "FSENDER001",
            "receiver": "RECEIVER001",
            "validator": "VALIDATOR001",
        },
    }

    test_app = create_app(overrides)

    # Sync module-level values computed at import time.
    app_module.ROLE_SIGNUP_KEYS = dict(test_app.config.get("ROLE_SIGNUP_KEYS") or {})
    app_module.ROLE_CONFIRMATION_CODES = dict(
        test_app.config.get("ROLE_CONFIRMATION_CODES") or {}
    )
    app_module.DEFAULT_SENDER_ASSIGNMENTS = int(
        test_app.config.get("DEFAULT_SENDER_ASSIGNMENTS")
        or app_module.DEFAULT_SENDER_ASSIGNMENTS
    )
    app_module.DEFAULT_RECEIVER_ASSIGNMENTS = int(
        test_app.config.get("DEFAULT_RECEIVER_ASSIGNMENTS")
        or app_module.DEFAULT_RECEIVER_ASSIGNMENTS
    )
    app_module.DEFAULT_VALIDATOR_ASSIGNMENTS = int(
        test_app.config.get("DEFAULT_VALIDATOR_ASSIGNMENTS")
        or app_module.DEFAULT_VALIDATOR_ASSIGNMENTS
    )

    yield test_app


@pytest.fixture()
def db(app):
    from ouvia.extensions import db as _db

    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()


@pytest.fixture()
def client(app, db):
    return app.test_client()


@pytest.fixture()
def logout(client):
    def _logout() -> None:
        client.get("/logout", follow_redirects=False)

    return _logout


@pytest.fixture()
def login(client, logout) -> Callable[..., None]:
    def _login(unique_id: str, password: str | None = None) -> None:
        logout()
        data = {"unique_id": unique_id}
        if password is not None:
            data["password"] = password
        resp = client.post("/login", data=data, follow_redirects=False)
        assert resp.status_code in (302, 303)

    return _login


@pytest.fixture()
def make_user(db) -> Callable[..., "User"]:
    from ouvia.models import User

    def _make_user(
        *,
        unique_id: str,
        role: str,
        password: str | None = None,
        gender: str | None = None,
        trans: str | None = None,
        accept_terms: bool = False,
    ) -> User:
        user = User(unique_id=unique_id, role=role, gender=gender, trans=trans)
        if password is not None:
            user.set_password(password)
        if accept_terms:
            user.mark_terms_accepted()
        db.session.add(user)
        db.session.commit()
        return user

    return _make_user


@pytest.fixture()
def seed_conversation(db) -> Callable[..., tuple["Conversation", list["Question"]]]:
    from ouvia.models import Conversation, Question

    def _seed(
        *,
        external_id: str = "C1",
        topic: str = "Topic",
        source: str = "Source",
        text: str = "Hello",
        question_count: int = 2,
    ) -> tuple[Conversation, list[Question]]:
        conv = Conversation(
            external_id=external_id,
            topic=topic,
            source=source,
            source_text=text,
        )
        db.session.add(conv)
        db.session.flush()

        questions: list[Question] = []
        for i in range(question_count):
            q = Question(
                conversation_id=conv.id,
                question_id=f"Q{i + 1}",
                question_text=f"Question {i + 1}?",
                answer=f"Answer {i + 1}",
                translated_question=f"Translated {i + 1}?",
            )
            db.session.add(q)
            questions.append(q)

        db.session.commit()
        return conv, questions

    return _seed


@pytest.fixture()
def make_audio_file() -> Callable[..., tuple[BytesIO, str]]:
    def _make(name: str = "recording.wav") -> tuple[BytesIO, str]:
        payload = b"RIFF" + b"\x00" * 64
        return BytesIO(payload), name

    return _make
