def test_export_snapshot_json_requires_admin():
    from ouvia import create_app
    from ouvia.extensions import db
    from ouvia.models import Conversation, ConversationEntry, User

    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        }
    )

    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(unique_id="admin1").first()
        if admin is None:
            admin = User(unique_id="admin1", role="admin")
            admin.set_password("secret")
            db.session.add(admin)
            db.session.commit()

        conv = Conversation.query.filter_by(external_id="C1").first()
        if conv is None:
            conv = Conversation(external_id="C1", topic="T")
            db.session.add(conv)
            db.session.commit()

        # Create a conversation entry; sender_id need not be a sender role for this test
        entry = ConversationEntry.query.filter_by(
            conversation_id=conv.id, sender_id=admin.id
        ).first()
        if entry is None:
            entry = ConversationEntry(conversation_id=conv.id, sender_id=admin.id)
            db.session.add(entry)
            db.session.commit()

        entry_id = entry.id

    with app.test_client() as client:
        # log in as admin
        resp = client.post("/login", data={"unique_id": "admin1", "password": "secret"})
        assert resp.status_code in (302, 200)

        r = client.get("/admin/export-snapshot")
        assert r.status_code == 404

        rjson = client.get("/admin/export-snapshot.json")
        assert rjson.status_code == 200
        assert rjson.headers.get("Content-Type", "").startswith("application/json")
        payload = rjson.get_json()
        assert isinstance(payload, list)
        # ensure our entry appears with expected keys
        found = False
        for item in payload:
            if item.get("entry_id") == entry_id:
                found = True
                assert "translation_model" in item
                assert "validator_corrections" in item
                assert "validator_question_total" in item
        assert found
