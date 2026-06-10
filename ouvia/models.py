from datetime import datetime, timezone

from flask import current_app
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .constants import ENTRY_STATUS_READY
from .extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    unique_id = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(32))
    trans = db.Column(db.String(32))
    password_hash = db.Column(db.String(255))
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    terms_accepted_at = db.Column(db.DateTime)
    instructions_read_at = db.Column(db.DateTime)
    confirmation_code = db.Column(db.String(8))
    login_nonce = db.Column(db.Integer, nullable=False, default=0)

    def set_password(self, password: str | None) -> None:
        if password:
            self.password_hash = generate_password_hash(password)
        else:
            self.password_hash = None

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    @property
    def requires_password(self) -> bool:
        return self.role == "admin" or bool(self.password_hash)

    @property
    def requires_terms_acceptance(self) -> bool:
        """Return whether terms acceptance is required for this user.

        This avoids importing application helpers (which would cause a circular
        import) by reading the AppSetting row directly using the configured key.
        """
        if self.role == "admin":
            return False
        try:
            key = (
                current_app.config.get("TERMS_ENFORCEMENT_SETTING_KEY")
                or "require_terms_acceptance"
            )
        except RuntimeError:
            # No active app context; default to False to avoid raising during
            # CLI or import-time operations.
            return False

        setting = AppSetting.query.filter_by(key=key).first()
        return (
            (setting.value == "1") if setting and setting.value is not None else False
        )

    @property
    def terms_accepted(self) -> bool:
        return bool(self.terms_accepted_at) or self.role == "admin"

    def mark_terms_accepted(self) -> None:
        self.terms_accepted_at = datetime.now(timezone.utc)

    @property
    def instructions_read(self) -> bool:
        return bool(self.instructions_read_at) or self.role == "admin"

    def mark_instructions_read(self) -> None:
        self.instructions_read_at = datetime.now(timezone.utc)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(64), nullable=False, unique=True)
    topic = db.Column(db.String(255))
    source = db.Column(db.String(255))
    source_text = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    questions = db.relationship(
        "Question",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Question(db.Model):
    __tablename__ = "questions"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id = db.Column(db.String(64))
    question_text = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text)
    translated_question = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    conversation = db.relationship("Conversation", back_populates="questions")


class ConversationEntry(db.Model):
    __tablename__ = "conversation_entries"
    __table_args__ = (
        db.UniqueConstraint(
            "conversation_id",
            "sender_id",
            name="uq_conversation_entry_sender",
        ),
        db.CheckConstraint(
            "status IN ('ready', 'recorded', 'translated', 'evaluated', 'validated', 'completed')",
            name="ck_conversation_entry_status",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    sender_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    receiver_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    validator_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    status = db.Column(db.String(20), nullable=False, default=ENTRY_STATUS_READY)
    translation_text = db.Column(db.Text, nullable=True)
    translation_model = db.Column(db.String(64), nullable=True)

    conversation = db.relationship("Conversation", backref="entries")
    sender = db.relationship("User", foreign_keys=[sender_id])
    receiver = db.relationship("User", foreign_keys=[receiver_id])
    validator = db.relationship("User", foreign_keys=[validator_id])
    receiver_responses = db.relationship(
        "ReceiverResponse",
        back_populates="conversation_entry",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    validator_evaluations = db.relationship(
        "ValidatorResponseEvaluation",
        back_populates="conversation_entry",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    final_sender_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    final_survey = db.relationship(
        "SenderFinalSurvey",
        back_populates="conversation_entry",
        cascade="all, delete-orphan",
        uselist=False,
    )
    final_sender = db.relationship("User", foreign_keys=[final_sender_id])


class Recording(db.Model):
    __tablename__ = "recordings"

    id = db.Column(db.Integer, primary_key=True)
    conversation_entry_id = db.Column(
        db.Integer,
        db.ForeignKey("conversation_entries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    stored_filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_mimetype = db.Column(db.String(64))
    file_size = db.Column(db.Integer)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    conversation_entry = db.relationship(
        "ConversationEntry",
        backref=db.backref("recording", uselist=False, cascade="all, delete-orphan"),
    )


class ReceiverResponse(db.Model):
    __tablename__ = "receiver_responses"
    __table_args__ = (
        db.UniqueConstraint(
            "conversation_entry_id",
            "question_id",
            name="uq_receiver_response_entry_question",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    conversation_entry_id = db.Column(
        db.Integer,
        db.ForeignKey("conversation_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id = db.Column(
        db.Integer,
        db.ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    response_text = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation_entry = db.relationship(
        "ConversationEntry",
        back_populates="receiver_responses",
    )
    question = db.relationship("Question")


class ValidatorReview(db.Model):
    __tablename__ = "validator_reviews"

    id = db.Column(db.Integer, primary_key=True)
    conversation_entry_id = db.Column(
        db.Integer,
        db.ForeignKey("conversation_entries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    validator_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    translation_score = db.Column(db.Integer, nullable=False)
    audio_text_mismatch = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation_entry = db.relationship(
        "ConversationEntry",
        backref=db.backref(
            "validator_review",
            uselist=False,
            cascade="all, delete-orphan",
        ),
    )
    validator = db.relationship("User")


class ValidatorResponseEvaluation(db.Model):
    __tablename__ = "validator_response_evaluations"
    __table_args__ = (
        db.UniqueConstraint(
            "conversation_entry_id",
            "question_id",
            name="uq_validator_eval_entry_question",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    conversation_entry_id = db.Column(
        db.Integer,
        db.ForeignKey("conversation_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id = db.Column(
        db.Integer,
        db.ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    is_correct = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation_entry = db.relationship(
        "ConversationEntry",
        back_populates="validator_evaluations",
    )
    question = db.relationship("Question")


class SenderFinalSurvey(db.Model):
    __tablename__ = "sender_final_surveys"
    __table_args__ = (
        db.CheckConstraint(
            "baseline_usefulness_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_baseline_usefulness_score",
        ),
        db.CheckConstraint(
            "baseline_satisfaction_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_baseline_satisfaction_score",
        ),
        db.CheckConstraint(
            "baseline_trust_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_baseline_trust_score",
        ),
        db.CheckConstraint(
            "baseline_reliance_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_baseline_reliance_score",
        ),
        db.CheckConstraint(
            "usefulness_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_usefulness_score",
        ),
        db.CheckConstraint(
            "satisfaction_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_satisfaction_score",
        ),
        db.CheckConstraint(
            "trust_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_trust_score",
        ),
        db.CheckConstraint(
            "reliance_score BETWEEN 1 AND 5",
            name="ck_sender_final_survey_reliance_score",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    conversation_entry_id = db.Column(
        db.Integer,
        db.ForeignKey("conversation_entries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    sender_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    baseline_usefulness_score = db.Column(db.Integer, nullable=True)
    baseline_satisfaction_score = db.Column(db.Integer, nullable=True)
    baseline_trust_score = db.Column(db.Integer, nullable=True)
    baseline_reliance_score = db.Column(db.Integer, nullable=True)
    usefulness_score = db.Column(db.Integer, nullable=True)
    satisfaction_score = db.Column(db.Integer, nullable=True)
    trust_score = db.Column(db.Integer, nullable=True)
    reliance_score = db.Column(db.Integer, nullable=True)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation_entry = db.relationship(
        "ConversationEntry",
        back_populates="final_survey",
    )
    sender = db.relationship("User")


class AppSetting(db.Model):
    __tablename__ = "app_settings"

    key = db.Column(db.String(128), primary_key=True)
    value = db.Column(db.Text, nullable=False)
