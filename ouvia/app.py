import csv
import json
import logging
import os
import random
import secrets
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
    url_for,
)
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user,
)
from sqlalchemy import and_, event, func, or_
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, selectinload
from werkzeug.utils import secure_filename

from .config import Config
from .extensions import db, login_manager, migrate
from .models import (
    AppSetting,
    Conversation,
    ConversationEntry,
    Question,
    ReceiverResponse,
    Recording,
    SenderFinalSurvey,
    User,
    ValidatorResponseEvaluation,
    ValidatorReview,
)

app = Flask(__name__)
app.config.from_object(Config)

# Configure comprehensive logging
# Use a logs directory and a timestamped filename for per-run logs
LOG_DIR = Path(app.root_path) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_startup_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_log_file = LOG_DIR / f"{_startup_ts}-ouvia.log"

# Formatter used for handlers
_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure root logger, add console and file handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(logging.INFO)
root_logger.addHandler(_console_handler)

# File handler (writes to timestamped file)
_file_handler = logging.FileHandler(str(_log_file), encoding="utf-8")
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.INFO)
root_logger.addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.info(f"Flask application initializing; logs will be written to {_log_file}")


# Initialize extensions with the application
db.init_app(app)
migrate.init_app(app, db)
login_manager.init_app(app)
login_manager.login_view = "login"

logger.info("Database and login manager initialized")


# Enable SQLite foreign key support (required for ON DELETE SET NULL to work)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@app.context_processor
def inject_config_values():
    # Expose a small set of important config values to templates by default
    return {
        # "UI_PRIMARY_COLOR": app.config.get("UI_PRIMARY_COLOR"),
        # "UI_SECONDARY_COLOR": app.config.get("UI_SECONDARY_COLOR"),
        # "UI_LIGHT_BG": app.config.get("UI_LIGHT_BG"),
        "RECORDINGS_DIR": app.config.get("RECORDINGS_DIR"),
        "ROLE_SIGNUP_KEYS": ROLE_SIGNUP_KEYS,
    }


from .constants import (
    ENTRY_STATUS_CHOICES,
    ENTRY_STATUS_COMPLETED,
    ENTRY_STATUS_EVALUATED,
    ENTRY_STATUS_LABELS,
    ENTRY_STATUS_READY,
    ENTRY_STATUS_RECORDED,
    ENTRY_STATUS_TRANSLATED,
    ENTRY_STATUS_VALIDATED,
)

ENTRY_STATUS_ORDER: list[str] = [
    ENTRY_STATUS_READY,
    ENTRY_STATUS_RECORDED,
    ENTRY_STATUS_TRANSLATED,
    ENTRY_STATUS_EVALUATED,
    ENTRY_STATUS_VALIDATED,
    ENTRY_STATUS_COMPLETED,
]

_ENTRY_STATUS_INDEX: dict[str, int] = {
    status: idx for idx, status in enumerate(ENTRY_STATUS_ORDER)
}


def _get_status_index(status: str) -> int:
    if status not in _ENTRY_STATUS_INDEX:
        raise ValueError(f"Unknown entry status: {status}")
    return _ENTRY_STATUS_INDEX[status]


def _entry_downgrade_impacts(current_status: str, target_status: str) -> list[str]:
    """Return human-readable impacts when downgrading from current -> target.

    Impacts are computed based on which lifecycle states are crossed.
    """

    current_idx = _get_status_index(current_status)
    target_idx = _get_status_index(target_status)
    if target_idx >= current_idx:
        return []

    impacts: list[str] = []
    if current_idx >= _get_status_index(
        ENTRY_STATUS_COMPLETED
    ) and target_idx < _get_status_index(ENTRY_STATUS_COMPLETED):
        impacts.append("Delete sender final survey")
    if current_idx >= _get_status_index(
        ENTRY_STATUS_VALIDATED
    ) and target_idx < _get_status_index(ENTRY_STATUS_VALIDATED):
        impacts.append("Delete validator review and per-question evaluations")
    if current_idx >= _get_status_index(
        ENTRY_STATUS_EVALUATED
    ) and target_idx < _get_status_index(ENTRY_STATUS_EVALUATED):
        impacts.append("Delete receiver responses")
    if current_idx >= _get_status_index(
        ENTRY_STATUS_TRANSLATED
    ) and target_idx < _get_status_index(ENTRY_STATUS_TRANSLATED):
        impacts.append("Clear translation text and model")
    if current_idx >= _get_status_index(
        ENTRY_STATUS_RECORDED
    ) and target_idx < _get_status_index(ENTRY_STATUS_RECORDED):
        impacts.append("Delete recording (DB + disk)")
    return impacts


def _delete_entry_recording_from_disk(recording: Recording) -> None:
    recordings_dir = _ensure_recordings_dir().resolve()

    try:
        stored_rel = Path(recording.stored_filename)
        file_path = (recordings_dir / stored_rel).resolve()
        if recordings_dir not in file_path.parents and file_path != recordings_dir:
            logger.warning(
                "Refusing to delete recording outside recordings dir: %s", file_path
            )
        elif file_path.exists():
            file_path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete recording file from disk: %s", exc)

    # Also remove the entry directory if present (recordings are stored under entry_<id>/...)
    try:
        entry_dir = (
            recordings_dir / f"entry_{recording.conversation_entry_id}"
        ).resolve()
        if recordings_dir not in entry_dir.parents and entry_dir != recordings_dir:
            logger.warning(
                "Refusing to delete entry dir outside recordings dir: %s", entry_dir
            )
        elif entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete recording directory from disk: %s", exc)


TERMS_ENFORCEMENT_SETTING_KEY = app.config.get(
    "TERMS_ENFORCEMENT_SETTING_KEY", "require_terms_acceptance"
)
ROLE_SIGNUP_KEYS = app.config.get("ROLE_SIGNUP_KEYS", {})
SENDER_GENDER_OPTIONS = app.config.get(
    "SENDER_GENDER_OPTIONS", ["man", "woman", "non-binary"]
)
SENDER_GENDER_LABELS = app.config.get("SENDER_GENDER_LABELS", {})

# Whether the participant is trans (current gender differs from assigned at birth).
# Prefer the new *_TRANS_* config keys, but fall back to the older *_SEX_* keys.
SENDER_TRANS_OPTIONS = app.config.get(
    "SENDER_TRANS_OPTIONS",
    app.config.get("SENDER_SEX_OPTIONS", ["yes", "no", "rather-not-say"]),
)
SENDER_TRANS_LABELS = app.config.get(
    "SENDER_TRANS_LABELS",
    app.config.get("SENDER_SEX_LABELS", {}),
)

# Defaults for number of assignments created during signup
DEFAULT_SENDER_ASSIGNMENTS = app.config.get("DEFAULT_SENDER_ASSIGNMENTS")
DEFAULT_RECEIVER_ASSIGNMENTS = app.config.get("DEFAULT_RECEIVER_ASSIGNMENTS")
DEFAULT_VALIDATOR_ASSIGNMENTS = app.config.get("DEFAULT_VALIDATOR_ASSIGNMENTS")
DEFAULT_FINAL_SENDER_ASSIGNMENTS = app.config.get("DEFAULT_FINAL_SENDER_ASSIGNMENTS")

# Role-specific confirmation codes (all users of a role get the same code)
ROLE_CONFIRMATION_CODES = app.config.get("ROLE_CONFIRMATION_CODES", {})

# Sender reassignment settings (persisted as AppSetting rows)
ENFORCE_SAME_GENDER_KEY = "enforce_same_gender"
MAINTENANCE_MODE_KEY = "maintenance_mode"


RECEIVER_UI_LOCALE_KEY = "receiver_ui_locale"
DEFAULT_RECEIVER_UI_LOCALE = "en"
LOCALES_DIR = Path(app.root_path) / "locales"
_RECEIVER_LOCALE_CACHE: dict[str, "ReceiverLocalization"] = {}


_SETTINGS_TABLE_READY = False


class ReceiverLocalization:
    def __init__(self, code: str, label: str, strings: dict[str, str]):
        self.code = code
        self.label = label
        self.strings = strings

    def text(self, key: str, default: str = "", **kwargs: Any) -> str:
        template = self.strings.get(key, default if default is not None else key)
        if kwargs and isinstance(template, str):
            try:
                return template.format(**kwargs)
            except KeyError:
                return template
        return template

    def raw(self, key: str, default: str = "") -> str:
        return self.strings.get(key, default if default is not None else key)

    def to_dict(self) -> dict[str, str]:
        return self.strings


VALID_USER_ROLES = {"sender", "receiver", "validator", "admin"}
ROLE_DISPLAY_ORDER = [
    ("admin", "Admin"),
    ("sender", "Sender"),
    ("receiver", "Receiver"),
    ("validator", "Validator"),
]
ROLE_LABELS = {role: label for role, label in ROLE_DISPLAY_ORDER}

REQUIRED_CONVERSATION_COLUMNS = {
    "conv_id",
    "topic",
    "source",
    "text",
    "qid",
    "question",
    "answer",
    "translated_question",
}

REQUIRED_USER_COLUMNS = {
    "unique_id",
    "role",
}

REQUIRED_CONVERSATION_ENTRY_COLUMNS = {
    "sender_id",
    "conv_id",
}

REQUIRED_RECEIVER_ASSIGNMENT_COLUMNS = {
    "conv_entry_id",
    "receiver_id",
}

REQUIRED_VALIDATOR_ASSIGNMENT_COLUMNS = {
    "conv_entry_id",
    "validator_id",
}


def _user_to_dict(user: "User", include_recordings: bool = False) -> dict[str, Any]:
    result = {
        "id": user.id,
        "unique_id": user.unique_id,
        "role": user.role,
        "role_label": ROLE_LABELS.get(user.role, user.role.title()),
        "gender": user.gender,
        "gender_label": (
            SENDER_GENDER_LABELS.get(user.gender, user.gender.title())
            if user.gender
            else None
        ),
        "trans": user.trans,
        "trans_label": (
            SENDER_TRANS_LABELS.get(user.trans, user.trans.title())
            if user.trans
            else None
        ),
        "terms_accepted": bool(user.terms_accepted_at) or user.role == "admin",
        "terms_accepted_at": (
            user.terms_accepted_at.isoformat() if user.terms_accepted_at else None
        ),
        "created_at": user.created_at.strftime("%Y-%m-%d %H:%M"),
        "has_password": bool(user.password_hash),
        "confirmation_code": user.confirmation_code,
    }

    if include_recordings:
        if user.role == "sender":
            total_entries = ConversationEntry.query.filter_by(sender_id=user.id).count()
            statuses_validated_or_below = [
                ENTRY_STATUS_READY,
                ENTRY_STATUS_RECORDED,
                ENTRY_STATUS_TRANSLATED,
                ENTRY_STATUS_EVALUATED,
                ENTRY_STATUS_VALIDATED,
            ]
            tasks_validated_or_below = ConversationEntry.query.filter(
                ConversationEntry.sender_id == user.id,
                ConversationEntry.status.in_(statuses_validated_or_below),
            ).count()
            tasks_completed = ConversationEntry.query.filter(
                ConversationEntry.sender_id == user.id,
                ConversationEntry.status == ENTRY_STATUS_COMPLETED,
            ).count()
            result["tasks_total"] = total_entries
            result["tasks_sender_validated_or_below"] = tasks_validated_or_below
            result["tasks_sender_completed"] = tasks_completed
        elif user.role == "receiver":
            total_entries = ConversationEntry.query.filter_by(receiver_id=user.id).count()
            statuses_evaluated_or_above = [
                ENTRY_STATUS_EVALUATED,
                ENTRY_STATUS_VALIDATED,
                ENTRY_STATUS_COMPLETED,
            ]
            tasks_evaluated = ConversationEntry.query.filter(
                ConversationEntry.receiver_id == user.id,
                ConversationEntry.status.in_(statuses_evaluated_or_above),
            ).count()
            result["tasks_total"] = total_entries
            result["tasks_receiver_evaluated"] = tasks_evaluated
        elif user.role == "validator":
            total_entries = ConversationEntry.query.filter_by(validator_id=user.id).count()
            statuses_validated_or_above = [
                ENTRY_STATUS_VALIDATED,
                ENTRY_STATUS_COMPLETED,
            ]
            tasks_validated = ConversationEntry.query.filter(
                ConversationEntry.validator_id == user.id,
                ConversationEntry.status.in_(statuses_validated_or_above),
            ).count()
            result["tasks_total"] = total_entries
            result["tasks_validator_validated"] = tasks_validated
        else:
            result["tasks_total"] = 0

    return result


def _get_user_task_stats(user_ids: list[int]) -> dict[int, dict[str, int]]:
    """Return per-user task metrics with a constant number of aggregate queries."""
    if not user_ids:
        return {}

    stats: dict[int, dict[str, int]] = {
        user_id: {
            "tasks_total": 0,
            "tasks_sender_validated_or_below": 0,
            "tasks_sender_completed": 0,
            "tasks_receiver_evaluated": 0,
            "tasks_validator_validated": 0,
        }
        for user_id in user_ids
    }

    sender_total_rows = (
        db.session.query(
            ConversationEntry.sender_id,
            func.count(ConversationEntry.id),
        )
        .filter(ConversationEntry.sender_id.in_(user_ids))
        .group_by(ConversationEntry.sender_id)
        .all()
    )
    for sender_id, count in sender_total_rows:
        if sender_id in stats:
            stats[sender_id]["tasks_total"] = int(count)

    sender_validated_or_below_rows = (
        db.session.query(
            ConversationEntry.sender_id,
            func.count(ConversationEntry.id),
        )
        .filter(
            ConversationEntry.sender_id.in_(user_ids),
            ConversationEntry.status.in_(
                [
                    ENTRY_STATUS_READY,
                    ENTRY_STATUS_RECORDED,
                    ENTRY_STATUS_TRANSLATED,
                    ENTRY_STATUS_EVALUATED,
                    ENTRY_STATUS_VALIDATED,
                ]
            ),
        )
        .group_by(ConversationEntry.sender_id)
        .all()
    )
    for sender_id, count in sender_validated_or_below_rows:
        if sender_id in stats:
            stats[sender_id]["tasks_sender_validated_or_below"] = int(count)

    sender_completed_rows = (
        db.session.query(
            ConversationEntry.sender_id,
            func.count(ConversationEntry.id),
        )
        .filter(
            ConversationEntry.sender_id.in_(user_ids),
            ConversationEntry.status == ENTRY_STATUS_COMPLETED,
        )
        .group_by(ConversationEntry.sender_id)
        .all()
    )
    for sender_id, count in sender_completed_rows:
        if sender_id in stats:
            stats[sender_id]["tasks_sender_completed"] = int(count)

    receiver_total_rows = (
        db.session.query(
            ConversationEntry.receiver_id,
            func.count(ConversationEntry.id),
        )
        .filter(ConversationEntry.receiver_id.in_(user_ids))
        .group_by(ConversationEntry.receiver_id)
        .all()
    )
    for receiver_id, count in receiver_total_rows:
        if receiver_id in stats:
            stats[receiver_id]["tasks_total"] = int(count)

    receiver_evaluated_rows = (
        db.session.query(
            ConversationEntry.receiver_id,
            func.count(ConversationEntry.id),
        )
        .filter(
            ConversationEntry.receiver_id.in_(user_ids),
            ConversationEntry.status.in_(
                [
                    ENTRY_STATUS_EVALUATED,
                    ENTRY_STATUS_VALIDATED,
                    ENTRY_STATUS_COMPLETED,
                ]
            ),
        )
        .group_by(ConversationEntry.receiver_id)
        .all()
    )
    for receiver_id, count in receiver_evaluated_rows:
        if receiver_id in stats:
            stats[receiver_id]["tasks_receiver_evaluated"] = int(count)

    validator_total_rows = (
        db.session.query(
            ConversationEntry.validator_id,
            func.count(ConversationEntry.id),
        )
        .filter(ConversationEntry.validator_id.in_(user_ids))
        .group_by(ConversationEntry.validator_id)
        .all()
    )
    for validator_id, count in validator_total_rows:
        if validator_id in stats:
            stats[validator_id]["tasks_total"] = int(count)

    validator_validated_rows = (
        db.session.query(
            ConversationEntry.validator_id,
            func.count(ConversationEntry.id),
        )
        .filter(
            ConversationEntry.validator_id.in_(user_ids),
            ConversationEntry.status.in_(
                [
                    ENTRY_STATUS_VALIDATED,
                    ENTRY_STATUS_COMPLETED,
                ]
            ),
        )
        .group_by(ConversationEntry.validator_id)
        .all()
    )
    for validator_id, count in validator_validated_rows:
        if validator_id in stats:
            stats[validator_id]["tasks_validator_validated"] = int(count)

    return stats


def _normalize_role(role_value: Optional[str]) -> Optional[str]:
    if role_value is None:
        return None
    if isinstance(role_value, str):
        candidate = role_value.strip().lower()
    else:
        candidate = str(role_value).strip().lower()
    return candidate if candidate in VALID_USER_ROLES else None


def _clean_password(password_value) -> Optional[str]:
    if password_value is None:
        return None
    if isinstance(password_value, str):
        cleaned = password_value.strip()
    else:
        cleaned = str(password_value).strip()
    return cleaned or None


def _resolve_user_by_unique_id(
    unique_id: Optional[str], cache: Optional[dict[str, Optional["User"]]] = None
) -> Optional["User"]:
    if not unique_id:
        return None

    if cache is not None and unique_id in cache:
        return cache[unique_id]

    user = User.query.filter_by(unique_id=unique_id).first()

    if cache is not None:
        cache[unique_id] = user

    return user


def _resolve_conversation_by_external_id(
    external_id: Optional[str],
    cache: Optional[dict[str, Optional[Conversation]]],
) -> Optional[Conversation]:
    if not external_id:
        return None

    if cache is not None and external_id in cache:
        return cache[external_id]

    conversation = Conversation.query.filter_by(external_id=external_id).first()

    if cache is not None:
        cache[external_id] = conversation

    return conversation


def get_post_login_redirect(user: "User") -> str:
    if user.role == "admin":
        return url_for("dashboard")
    if user.requires_terms_acceptance and not user.terms_accepted:
        return url_for("terms")
    return url_for("participant_home")


@login_manager.user_loader
def load_user(user_id: str):
    user = db.session.get(User, int(user_id))
    if user:
        logger.debug(f"User loaded: {user.unique_id} (role: {user.role})")
    return user


def _session_get_or_404(model: type, ident: Any):
    obj = db.session.get(model, ident)
    if obj is None:
        abort(404)
    return obj


@app.before_request
def enforce_maintenance_mode():
    if not is_maintenance_enabled():
        return

    endpoint = request.endpoint or ""
    endpoint_root = endpoint.split(".")[0] if endpoint else ""

    if endpoint_root == "static":
        return

    if current_user.is_authenticated:
        if current_user.role == "admin":
            return
        if endpoint in {"maintenance_page", "logout"}:
            return
        logger.info(
            "Redirecting user %s to maintenance page",
            current_user.unique_id,
        )
        return redirect(url_for("maintenance_page"))

    if endpoint in {"login", "maintenance_page"}:
        return

    return redirect(url_for("maintenance_page"))


@app.before_request
def enforce_terms_acceptance():
    if not current_user.is_authenticated:
        return

    if not is_terms_enforcement_enabled():
        return

    if current_user.role == "admin" or current_user.terms_accepted:
        return

    endpoint = request.endpoint or ""
    allowed_endpoints = {"terms", "logout", "static"}

    if endpoint and endpoint.split(".")[0] in allowed_endpoints:
        return

    logger.info(f"Redirecting user {current_user.unique_id} to terms acceptance page")
    return redirect(url_for("terms"))


@app.before_request
def enforce_session_nonce():
    if not current_user.is_authenticated:
        return

    if current_user.role == "admin":
        return

    endpoint = request.endpoint or ""
    if endpoint in {"login", "logout", "static"}:
        return

    user_nonce = current_user.login_nonce
    session_nonce = session.get("login_nonce")

    if session_nonce is None:
        # Legacy session: no nonce stored yet.
        if user_nonce == 0:
            # Nonce hasn't been bumped yet — store and allow.
            session["login_nonce"] = 0
            return
        # Nonce was bumped but session predates the feature — force logout.
        logger.info(
            "Logging out user %s: legacy session after nonce bump",
            current_user.unique_id,
        )
        logout_user()
        flash("Please log in again to continue.", "warning")
        return redirect(request.url)

    if session_nonce != user_nonce:
        logger.info(
            "Logging out user %s: session nonce %s != user nonce %s",
            current_user.unique_id,
            session_nonce,
            user_nonce,
        )
        logout_user()
        flash("Your session has expired. Please log in again.", "warning")
        return redirect(request.url)


def handle_conversation_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.filename == "":
        logger.warning("Conversation upload attempted without file")
        flash("Please choose a CSV file to upload.", "warning")
        return None

    logger.info(f"Processing conversation upload: {uploaded_file.filename}")
    try:
        dataframe = pd.read_csv(uploaded_file)
        logger.info(f"CSV file parsed successfully, {len(dataframe)} rows found")
    except Exception as exc:  # noqa: BLE001 - surface issue to user
        logger.error(f"Failed to read CSV file {uploaded_file.filename}: {exc}")
        flash(f"Failed to read CSV file: {exc}", "danger")
        return None

    missing_columns = REQUIRED_CONVERSATION_COLUMNS - set(dataframe.columns)
    if missing_columns:
        formatted_missing = ", ".join(sorted(missing_columns))
        logger.error(f"CSV missing required columns: {formatted_missing}")
        flash(
            f"CSV missing required columns: {formatted_missing}",
            "danger",
        )
        return None

    stats = import_conversations_from_dataframe(dataframe)

    try:
        db.session.commit()
        logger.info(f"Conversations imported successfully: {stats}")
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(f"Database error while saving imported conversations: {exc}")
        flash("Could not save conversations due to a database error.", "danger")
        return None

    flash("Conversations imported successfully.", "success")
    return stats


def import_conversations_from_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    stats: dict[str, int] = {
        "conversations_created": 0,
        "conversations_updated": 0,
        "conversations_skipped": 0,
        "questions_created": 0,
        "questions_skipped": 0,
    }

    grouped = dataframe.groupby("conv_id", dropna=False)
    logger.info(f"Importing {len(grouped)} conversations from dataframe")

    for raw_conv_id, group in grouped:
        conversation_key = _clean_value(raw_conv_id)

        if not conversation_key:
            stats["conversations_skipped"] += 1
            logger.warning(f"Skipping conversation with empty/invalid ID")
            continue

        conversation = Conversation.query.filter_by(
            external_id=conversation_key
        ).first()

        if conversation is None:
            conversation = Conversation(external_id=conversation_key)
            db.session.add(conversation)
            stats["conversations_created"] += 1
            logger.info(f"Creating new conversation: {conversation_key}")
        else:
            stats["conversations_updated"] += 1
            logger.debug(f"Updating existing conversation: {conversation_key}")

        first_row = group.iloc[0]
        conversation.topic = _clean_value(first_row.get("topic"))
        conversation.source = _clean_value(first_row.get("source"))
        conversation.source_text = _clean_value(first_row.get("text"))

        existing_signatures: set[tuple[str, str]] = {
            ((question.question_id or ""), (question.question_text or ""))
            for question in conversation.questions
        }

        for _, row in group.iterrows():
            question_text = _clean_value(row.get("question"))
            if not question_text:
                stats["questions_skipped"] += 1
                continue

            question_id = _clean_value(row.get("qid"))
            signature = ((question_id or ""), question_text)

            if signature in existing_signatures:
                stats["questions_skipped"] += 1
                continue

            question = Question(
                question_id=question_id,
                question_text=question_text,
                answer=_clean_value(row.get("answer")),
                translated_question=_clean_value(row.get("translated_question")),
            )

            conversation.questions.append(question)
            existing_signatures.add(signature)
            stats["questions_created"] += 1

    return stats


def _clean_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if pd.isna(value):
        return None
    stripped = str(value).strip()
    return stripped or None


def _to_int(value) -> Optional[int]:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        return None


def _stratified_sample_by_key(items: list[Any], k: int, key_fn) -> list[Any]:
    """Return up to k items using stratified sampling over key_fn.

    Guarantees no more than k items; if population <= k, returns a shuffled copy.
    """

    if k <= 0:
        return []

    if len(items) <= k:
        shuffled = list(items)
        random.shuffle(shuffled)
        return shuffled

    groups: dict[Any, list[Any]] = {}
    for item in items:
        key = key_fn(item) or ""
        groups.setdefault(key, []).append(item)

    total = len(items)
    allocation: dict[Any, int] = {}
    remainders: list[tuple[float, Any]] = []

    for key, group in groups.items():
        exact = (k * len(group)) / total
        base = int(exact)
        base = min(base, len(group))
        allocation[key] = base
        remainders.append((exact - base, key))

    remaining = k - sum(allocation.values())
    remainders.sort(key=lambda x: x[0], reverse=True)

    for _, key in remainders:
        if remaining <= 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            remaining -= 1

    if remaining > 0:
        for key, group in groups.items():
            if remaining <= 0:
                break
            space = len(group) - allocation[key]
            if space <= 0:
                continue
            take = min(space, remaining)
            allocation[key] += take
            remaining -= take

    selected: list[Any] = []
    for key, group in groups.items():
        take = allocation.get(key, 0)
        if take > 0:
            selected.extend(random.sample(group, take))

    if len(selected) > k:
        random.shuffle(selected)
        selected = selected[:k]

    return selected


def _assign_sender_tasks(
    user: "User", *, batch_size: int = DEFAULT_SENDER_ASSIGNMENTS
) -> Optional[str]:
    logger.info(
        f"Assigning sender tasks to user {user.unique_id} (gender: {user.gender})"
    )

    if not user.gender:
        logger.warning(
            f"Cannot assign sender tasks to {user.unique_id}: gender not specified"
        )
        return "signup.error.sender_gender_required"

    assigned_same_gender = (
        db.session.query(ConversationEntry.conversation_id)
        .join(User, ConversationEntry.sender_id == User.id)
        .filter(User.gender == user.gender)
    )

    available_conversations = Conversation.query.filter(
        ~Conversation.id.in_(assigned_same_gender)
    ).all()

    if len(available_conversations) < batch_size:
        logger.error(
            f"Not enough conversations available for sender {user.unique_id}: need {batch_size}, found {len(available_conversations)}"
        )
        return "signup.error.not_enough_sender_entries"

    selected = _stratified_sample_by_key(
        available_conversations, batch_size, lambda conv: conv.source or "unknown"
    )

    if len(selected) < batch_size:
        logger.error(
            f"Stratified sampling failed for sender {user.unique_id}: need {batch_size}, selected {len(selected)}"
        )
        return "signup.error.not_enough_sender_entries"

    for conversation in selected:
        db.session.add(
            ConversationEntry(conversation_id=conversation.id, sender_id=user.id)
        )

    logger.info(f"Assigned {len(selected)} sender tasks to user {user.unique_id}")
    return None


def _assign_receiver_tasks(
    user: "User", *, batch_size: int = DEFAULT_RECEIVER_ASSIGNMENTS
) -> Optional[str]:
    logger.info(f"Assigning receiver tasks to user {user.unique_id}")

    # Only assign entries that are in 'translated' status and have no receiver
    available_entries = (
        ConversationEntry.query.options(joinedload(ConversationEntry.conversation))
        .filter(
            ConversationEntry.receiver_id.is_(None),
            ConversationEntry.status == ENTRY_STATUS_TRANSLATED,
        )
        .all()
    )

    if len(available_entries) < batch_size:
        logger.error(
            f"Not enough translated entries for receiver {user.unique_id}: need {batch_size}, found {len(available_entries)}"
        )
        return "signup.error.not_enough_receiver_entries"

    selected = _stratified_sample_by_key(
        available_entries,
        batch_size,
        lambda entry: (entry.conversation.source if entry.conversation else "unknown"),
    )

    if len(selected) < batch_size:
        logger.error(
            f"Stratified sampling failed for receiver {user.unique_id}: need {batch_size}, selected {len(selected)}"
        )
        return "signup.error.not_enough_receiver_entries"

    for entry in selected:
        entry.receiver_id = user.id

    logger.info(f"Assigned {len(selected)} receiver tasks to user {user.unique_id}")
    return None


def _assign_validator_tasks(
    user: "User", *, batch_size: int = DEFAULT_VALIDATOR_ASSIGNMENTS
) -> Optional[str]:
    logger.info(f"Assigning validator tasks to user {user.unique_id}")

    available_entries = (
        ConversationEntry.query.options(joinedload(ConversationEntry.conversation))
        .filter(
            ConversationEntry.validator_id.is_(None),
            ConversationEntry.status == ENTRY_STATUS_EVALUATED,
        )
        .all()
    )

    if len(available_entries) < batch_size:
        logger.error(
            f"Not enough entries for validator {user.unique_id}: need {batch_size}, found {len(available_entries)}"
        )
        return "signup.error.not_enough_validator_entries"

    selected = _stratified_sample_by_key(
        available_entries,
        batch_size,
        lambda entry: (entry.conversation.source if entry.conversation else "unknown"),
    )

    if len(selected) < batch_size:
        logger.error(
            f"Stratified sampling failed for validator {user.unique_id}: need {batch_size}, selected {len(selected)}"
        )
        return "signup.error.not_enough_validator_entries"

    for entry in selected:
        entry.validator_id = user.id

    logger.info(f"Assigned {len(selected)} validator tasks to user {user.unique_id}")
    return None


def _find_eligible_senders_for_final_sender(
    new_gender: Optional[str],
) -> list["User"]:
    """Return senders whose *all* assigned entries are in 'validated' status
    and have no final_sender assigned yet.

    When ``enforce_same_gender`` is enabled, only senders with a matching gender
    are returned.
    """
    enforce_gender = is_enforce_same_gender_enabled()

    # Subquery: senders that have at least one entry NOT in 'validated'
    senders_with_non_validated = (
        db.session.query(ConversationEntry.sender_id)
        .filter(ConversationEntry.status != ENTRY_STATUS_VALIDATED)
        .distinct()
        .subquery()
    )

    # Senders who have at least one entry (any status) — we only want those with assignments
    senders_with_entries = (
        db.session.query(ConversationEntry.sender_id)
        .distinct()
        .subquery()
    )

    # Senders whose entries already have a final_sender assigned
    senders_with_final_sender = (
        db.session.query(ConversationEntry.sender_id)
        .filter(ConversationEntry.final_sender_id.isnot(None))
        .distinct()
        .subquery()
    )

    query = User.query.filter(
        User.role == "sender",
        User.id.in_(db.session.query(senders_with_entries.c.sender_id)),
        ~User.id.in_(db.session.query(senders_with_non_validated.c.sender_id)),
        ~User.id.in_(db.session.query(senders_with_final_sender.c.sender_id)),
    )

    if enforce_gender and new_gender:
        query = query.filter(User.gender == new_gender)

    return query.all()


def _assign_final_sender_entries(
    old_sender: "User", new_final_sender: "User"
) -> int:
    """Assign *new_final_sender* as the final_sender on all entries of *old_sender*.

    Returns the number of entries updated.
    """
    entries = ConversationEntry.query.filter_by(sender_id=old_sender.id).all()
    count = 0

    for entry in entries:
        entry.final_sender_id = new_final_sender.id
        count += 1

    logger.info(
        f"Assigned final sender {new_final_sender.unique_id} to {count} entries "
        f"from original sender {old_sender.unique_id}"
    )
    return count


def _assign_final_sender_tasks(
    user: "User", *, batch_size: int = DEFAULT_FINAL_SENDER_ASSIGNMENTS
) -> Optional[str]:
    """Assign validated entries to a new final sender.

    Finds an eligible original sender whose entries are all validated and have no
    final_sender yet, and assigns *user* as the final_sender on those entries.

    Returns ``None`` on success or an error key string on failure.
    """
    logger.info(
        f"Assigning final sender tasks to user {user.unique_id} "
        f"(gender: {user.gender})"
    )

    eligible = _find_eligible_senders_for_final_sender(user.gender)
    if not eligible:
        logger.error(
            f"No eligible sender found for final sender assignment to "
            f"{user.unique_id} (gender={user.gender})"
        )
        return "signup.error.not_enough_final_sender_entries"

    old_sender = random.choice(eligible)
    logger.info(
        f"Selected sender {old_sender.unique_id} (gender={old_sender.gender}) "
        f"for final sender assignment to {user.unique_id}"
    )

    assigned = _assign_final_sender_entries(old_sender, user)
    if assigned == 0:
        logger.warning(
            f"Final sender assignment yielded 0 entries for sender "
            f"{old_sender.unique_id}"
        )
        return "signup.error.not_enough_final_sender_entries"

    return None


def handle_user_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.filename == "":
        logger.warning("User upload attempted without file")
        flash("Please choose a CSV file to upload.", "warning")
        return None

    logger.info(f"Processing user upload: {uploaded_file.filename}")
    try:
        dataframe = pd.read_csv(uploaded_file)
        logger.info(f"User CSV file parsed successfully, {len(dataframe)} rows found")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to read user CSV file {uploaded_file.filename}: {exc}")
        flash(f"Failed to read CSV file: {exc}", "danger")
        return None

    missing_columns = REQUIRED_USER_COLUMNS - set(dataframe.columns)
    if missing_columns:
        formatted_missing = ", ".join(sorted(missing_columns))
        logger.error(f"User CSV missing required columns: {formatted_missing}")
        flash(
            f"CSV missing required columns: {formatted_missing}",
            "danger",
        )
        return None

    stats = import_users_from_dataframe(dataframe)

    try:
        db.session.commit()
        logger.info(f"Users imported successfully: {stats}")
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(f"Database error while saving imported users: {exc}")
        flash("Could not save users due to a database error.", "danger")
        return None

    flash("Users imported successfully.", "success")
    return stats


def import_users_from_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    stats: dict[str, int] = {
        "users_created": 0,
        "users_updated": 0,
        "users_skipped": 0,
        "invalid_rows": 0,
    }

    seen_ids: set[str] = set()

    for _, row in dataframe.iterrows():
        unique_id = _clean_value(row.get("unique_id"))
        normalized_role = _normalize_role(row.get("role"))

        if not unique_id or not normalized_role:
            stats["invalid_rows"] += 1
            continue

        if unique_id in seen_ids:
            stats["users_skipped"] += 1
            continue

        seen_ids.add(unique_id)

        user = User.query.filter_by(unique_id=unique_id).first()

        if user is None:
            user = User(unique_id=unique_id, role=normalized_role)
            if normalized_role == "admin" and not user.password_hash:
                user.terms_accepted_at = datetime.now(timezone.utc)
            db.session.add(user)
            stats["users_created"] += 1
        else:
            previous_role = user.role
            user.role = normalized_role
            if normalized_role != "admin" and previous_role == "admin":
                user.terms_accepted_at = None
            stats["users_updated"] += 1

    return stats


def handle_conversation_entry_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.filename == "":
        logger.warning("Conversation entry upload attempted without file")
        flash("Please choose a CSV file to upload.", "warning")
        return None

    logger.info(f"Processing conversation entry upload: {uploaded_file.filename}")
    try:
        dataframe = pd.read_csv(uploaded_file)
        logger.info(
            f"Conversation entry CSV parsed successfully, {len(dataframe)} rows found"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"Failed to read conversation entry CSV {uploaded_file.filename}: {exc}"
        )
        flash(f"Failed to read CSV file: {exc}", "danger")
        return None

    missing_columns = REQUIRED_CONVERSATION_ENTRY_COLUMNS - set(dataframe.columns)
    if missing_columns:
        formatted_missing = ", ".join(sorted(missing_columns))
        logger.error(
            f"Conversation entry CSV missing required columns: {formatted_missing}"
        )
        flash(
            f"CSV missing required columns: {formatted_missing}",
            "danger",
        )
        return None

    stats = import_conversation_entries_from_dataframe(dataframe)

    if stats["entries_created"] == 0:
        detail_parts: list[str] = []
        if stats["missing_conversations"]:
            detail_parts.append(
                f"{stats['missing_conversations']} row(s) referenced conversation IDs that were not found."
            )
        if stats["missing_users"]:
            detail_parts.append(
                f"{stats['missing_users']} row(s) referenced sender IDs that do not exist."
            )
        if stats["role_mismatch"]:
            detail_parts.append(
                f"{stats['role_mismatch']} row(s) were skipped because the sender does not have the 'sender' role."
            )
        if stats["entries_skipped_existing"]:
            detail_parts.append(
                f"{stats['entries_skipped_existing']} row(s) matched existing conversation entries and were ignored."
            )
        if stats["duplicate_rows"]:
            detail_parts.append(
                f"{stats['duplicate_rows']} duplicate row(s) were ignored."
            )
        if stats["invalid_rows"]:
            detail_parts.append(
                f"{stats['invalid_rows']} row(s) were missing required values."
            )

        if not detail_parts:
            detail_parts.append("Please verify the CSV contents and try again.")

        flash(
            "No conversation entries were imported. " + " ".join(detail_parts),
            "danger",
        )
        return stats

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception("Error while saving conversation entries", exc_info=exc)
        flash("Could not save conversation entries due to a database error.", "danger")
        return None

    flash("Conversation entries imported successfully.", "success")

    if stats["entries_skipped_existing"]:
        flash(
            f"{stats['entries_skipped_existing']} row(s) matched existing conversation entries and were ignored.",
            "info",
        )

    if stats["duplicate_rows"]:
        flash(
            f"{stats['duplicate_rows']} duplicate row(s) were ignored.",
            "info",
        )

    if stats["role_mismatch"]:
        flash(
            f"{stats['role_mismatch']} row(s) were skipped because the sender does not have the 'sender' role.",
            "warning",
        )

    return stats


def import_conversation_entries_from_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "entries_created": 0,
        "entries_skipped_existing": 0,
        "invalid_rows": 0,
        "missing_users": 0,
        "missing_conversations": 0,
        "duplicate_rows": 0,
        "role_mismatch": 0,
    }

    user_cache: dict[str, Optional[User]] = {}
    conversation_cache: dict[str, Optional[Conversation]] = {}
    seen_signatures: set[tuple[int, int]] = set()

    for _, row in dataframe.iterrows():
        sender_unique_id = _clean_value(row.get("sender_id"))
        conversation_external_id = _clean_value(row.get("conv_id"))

        if not sender_unique_id or not conversation_external_id:
            stats["invalid_rows"] += 1
            continue

        conversation = _resolve_conversation_by_external_id(
            conversation_external_id,
            conversation_cache,
        )
        if conversation is None:
            stats["missing_conversations"] += 1
            continue

        sender = _resolve_user_by_unique_id(sender_unique_id, user_cache)
        if sender is None:
            stats["missing_users"] += 1
            continue

        if sender.role != "sender":
            stats["role_mismatch"] += 1
            continue

        signature = (conversation.id, sender.id)
        if signature in seen_signatures:
            stats["duplicate_rows"] += 1
            continue

        seen_signatures.add(signature)

        existing_entry = ConversationEntry.query.filter_by(
            conversation_id=conversation.id,
            sender_id=sender.id,
        ).first()

        if existing_entry:
            stats["entries_skipped_existing"] += 1
            continue

        entry = ConversationEntry(
            conversation_id=conversation.id,
            sender_id=sender.id,
        )
        db.session.add(entry)
        stats["entries_created"] += 1

    return stats


def import_receiver_assignments_from_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "assignments_applied": 0,
        "assignments_updated": 0,
        "already_assigned": 0,
        "invalid_rows": 0,
        "missing_entries": 0,
        "missing_users": 0,
        "role_mismatch": 0,
        "duplicate_rows": 0,
    }

    entry_cache: dict[int, Optional[ConversationEntry]] = {}
    user_cache: dict[str, Optional[User]] = {}
    seen_pairs: set[tuple[int, str]] = set()

    for _, row in dataframe.iterrows():
        entry_id = _to_int(row.get("conv_entry_id"))
        receiver_unique_id = _clean_value(row.get("receiver_id"))

        if entry_id is None or not receiver_unique_id:
            stats["invalid_rows"] += 1
            continue

        pair_key = (entry_id, receiver_unique_id)
        if pair_key in seen_pairs:
            stats["duplicate_rows"] += 1
            continue
        seen_pairs.add(pair_key)

        entry = entry_cache.get(entry_id)
        if entry is None:
            entry = db.session.get(ConversationEntry, entry_id)
            entry_cache[entry_id] = entry

        if entry is None:
            stats["missing_entries"] += 1
            continue

        receiver = _resolve_user_by_unique_id(receiver_unique_id, user_cache)
        if receiver is None:
            stats["missing_users"] += 1
            continue

        if receiver.role != "receiver":
            stats["role_mismatch"] += 1
            continue

        if entry.receiver_id == receiver.id:
            stats["already_assigned"] += 1
            continue

        if entry.receiver_id and entry.receiver_id != receiver.id:
            entry.receiver_id = receiver.id
            stats["assignments_updated"] += 1
        else:
            entry.receiver_id = receiver.id
            stats["assignments_applied"] += 1

    return stats


def import_validator_assignments_from_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "assignments_applied": 0,
        "assignments_updated": 0,
        "already_assigned": 0,
        "invalid_rows": 0,
        "missing_entries": 0,
        "missing_users": 0,
        "role_mismatch": 0,
        "duplicate_rows": 0,
    }

    entry_cache: dict[int, Optional[ConversationEntry]] = {}
    user_cache: dict[str, Optional[User]] = {}
    seen_pairs: set[tuple[int, str]] = set()

    for _, row in dataframe.iterrows():
        entry_id = _to_int(row.get("conv_entry_id"))
        validator_unique_id = _clean_value(row.get("validator_id"))

        if entry_id is None or not validator_unique_id:
            stats["invalid_rows"] += 1
            continue

        pair_key = (entry_id, validator_unique_id)
        if pair_key in seen_pairs:
            stats["duplicate_rows"] += 1
            continue
        seen_pairs.add(pair_key)

        entry = entry_cache.get(entry_id)
        if entry is None:
            entry = db.session.get(ConversationEntry, entry_id)
            entry_cache[entry_id] = entry

        if entry is None:
            stats["missing_entries"] += 1
            continue

        validator = _resolve_user_by_unique_id(validator_unique_id, user_cache)
        if validator is None:
            stats["missing_users"] += 1
            continue

        if validator.role != "validator":
            stats["role_mismatch"] += 1
            continue

        if entry.validator_id == validator.id:
            stats["already_assigned"] += 1
            continue

        if entry.validator_id and entry.validator_id != validator.id:
            entry.validator_id = validator.id
            stats["assignments_updated"] += 1
        else:
            entry.validator_id = validator.id
            stats["assignments_applied"] += 1

    return stats


def _format_assignment_details(stats: dict[str, int]) -> list[str]:
    details: list[str] = []
    if stats.get("missing_entries"):
        details.append(
            f"{stats['missing_entries']} row(s) referenced conversation-entry IDs that were not found."
        )
    if stats.get("missing_users"):
        details.append(
            f"{stats['missing_users']} row(s) referenced user IDs that do not exist."
        )
    if stats.get("role_mismatch"):
        details.append(
            f"{stats['role_mismatch']} row(s) were skipped because the user does not match the required role."
        )
    if stats.get("duplicate_rows"):
        details.append(
            f"{stats['duplicate_rows']} duplicate row(s) were ignored in the CSV."
        )
    if stats.get("already_assigned"):
        details.append(
            f"{stats['already_assigned']} row(s) already matched the current assignment and were left unchanged."
        )
    if stats.get("invalid_rows"):
        details.append(f"{stats['invalid_rows']} row(s) were missing required values.")
    return details


def handle_receiver_assignment_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.filename == "":
        logger.warning("Receiver assignment upload attempted without file")
        flash("Please choose a CSV file to upload.", "warning")
        return None

    logger.info(f"Processing receiver assignment upload: {uploaded_file.filename}")
    try:
        dataframe = pd.read_csv(uploaded_file)
        logger.info(
            f"Receiver assignment CSV parsed successfully, {len(dataframe)} rows found"
        )
    except Exception as exc:  # noqa: BLE001 - user-facing message
        logger.error(
            f"Failed to read receiver assignment CSV {uploaded_file.filename}: {exc}"
        )
        flash(f"Failed to read CSV file: {exc}", "danger")
        return None

    missing_columns = REQUIRED_RECEIVER_ASSIGNMENT_COLUMNS - set(dataframe.columns)
    if missing_columns:
        formatted_missing = ", ".join(sorted(missing_columns))
        logger.error(
            f"Receiver assignment CSV missing required columns: {formatted_missing}"
        )
        flash(
            f"CSV missing required columns: {formatted_missing}",
            "danger",
        )
        return None

    stats = import_receiver_assignments_from_dataframe(dataframe)
    applied = stats["assignments_applied"] + stats["assignments_updated"]

    if applied == 0:
        details = _format_assignment_details(stats)
        if not details:
            details.append("Please verify the CSV contents and try again.")
        flash(
            "No receiver assignments were applied. " + " ".join(details),
            "danger",
        )
        return stats

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception(
            "Error while saving receiver assignments",
            exc_info=exc,
        )
        flash("Could not save receiver assignments due to a database error.", "danger")
        return None

    flash(
        "Receiver assignments updated successfully ("
        + f"{stats['assignments_applied']} new, {stats['assignments_updated']} reassigned).",
        "success",
    )

    for condition, message, level in [
        (
            stats["missing_users"],
            "User references were missing from the CSV.",
            "warning",
        ),
        (
            stats["missing_entries"],
            "Some rows referred to unknown conversation-entry IDs.",
            "warning",
        ),
        (
            stats["role_mismatch"],
            "Rows with incorrect roles were skipped.",
            "warning",
        ),
        (
            stats["duplicate_rows"],
            "Duplicate rows in the CSV were ignored.",
            "info",
        ),
        (
            stats["already_assigned"],
            "Rows that already matched the current assignment were left unchanged.",
            "info",
        ),
    ]:
        if condition:
            flash(message, level)

    return stats


def handle_validator_assignment_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.filename == "":
        logger.warning("Validator assignment upload attempted without file")
        flash("Please choose a CSV file to upload.", "warning")
        return None

    logger.info(f"Processing validator assignment upload: {uploaded_file.filename}")
    try:
        dataframe = pd.read_csv(uploaded_file)
        logger.info(
            f"Validator assignment CSV parsed successfully, {len(dataframe)} rows found"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"Failed to read validator assignment CSV {uploaded_file.filename}: {exc}"
        )
        flash(f"Failed to read CSV file: {exc}", "danger")
        return None

    missing_columns = REQUIRED_VALIDATOR_ASSIGNMENT_COLUMNS - set(dataframe.columns)
    if missing_columns:
        formatted_missing = ", ".join(sorted(missing_columns))
        logger.error(
            f"Validator assignment CSV missing required columns: {formatted_missing}"
        )
        flash(
            f"CSV missing required columns: {formatted_missing}",
            "danger",
        )
        return None

    stats = import_validator_assignments_from_dataframe(dataframe)
    applied = stats["assignments_applied"] + stats["assignments_updated"]

    if applied == 0:
        details = _format_assignment_details(stats)
        if not details:
            details.append("Please verify the CSV contents and try again.")
        flash(
            "No validator assignments were applied. " + " ".join(details),
            "danger",
        )
        return stats

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception(
            "Error while saving validator assignments",
            exc_info=exc,
        )
        flash("Could not save validator assignments due to a database error.", "danger")
        return None

    flash(
        "Validator assignments updated successfully ("
        + f"{stats['assignments_applied']} new, {stats['assignments_updated']} reassigned).",
        "success",
    )

    for condition, message, level in [
        (
            stats["missing_users"],
            "User references were missing from the CSV.",
            "warning",
        ),
        (
            stats["missing_entries"],
            "Some rows referred to unknown conversation-entry IDs.",
            "warning",
        ),
        (
            stats["role_mismatch"],
            "Rows with incorrect roles were skipped.",
            "warning",
        ),
        (
            stats["duplicate_rows"],
            "Duplicate rows in the CSV were ignored.",
            "info",
        ),
        (
            stats["already_assigned"],
            "Rows that already matched the current assignment were left unchanged.",
            "info",
        ),
    ]:
        if condition:
            flash(message, level)

    return stats


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(get_post_login_redirect(current_user))

    role_param = (request.args.get("role") or request.form.get("role") or "").lower()
    authkey_param = request.args.get("authkey") or request.form.get("authkey") or ""

    expected_key = ROLE_SIGNUP_KEYS.get(role_param)
    if not expected_key:
        abort(404)
    if authkey_param != expected_key:
        abort(403)

    prolific_id = (request.form.get("prolific_id") or "").strip()
    gender = (request.form.get("gender") or "").strip().lower()
    trans = (request.form.get("trans") or "").strip().lower()
    accepted_terms = request.form.get("accept_terms") == "on"
    error = None

    # Get localization for receivers (senders/validators use English)
    locale = (
        get_receiver_locale() if role_param == "receiver" else get_receiver_locale("en")
    )

    def _localize_error(error_key: str) -> str:
        """Translate error key to localized message, or return as-is if not a key."""
        if locale and error_key.startswith("signup."):
            return locale.text(error_key, default=error_key)
        return error_key

    if request.method == "POST":
        logger.info(
            f"Signup attempt for role {role_param}, prolific_id: {prolific_id or '<empty>'}"
        )
        if not accepted_terms:
            error = _localize_error("signup.error.terms_required")
            logger.warning(f"Signup failed: terms not accepted for {prolific_id}")
        elif not prolific_id:
            error = _localize_error("signup.error.prolific_id_required")
            logger.warning(
                f"Signup failed: prolific_id not provided for role {role_param}"
            )
        elif role_param == "sender" and (
            gender not in SENDER_GENDER_OPTIONS or trans not in SENDER_TRANS_OPTIONS
        ):
            error = _localize_error("signup.error.gender_trans_required")
            logger.warning(
                f"Signup failed: invalid gender/trans for sender {prolific_id} (gender={gender}, trans={trans})"
            )
        elif role_param == "final_sender" and (
            gender not in SENDER_GENDER_OPTIONS or trans not in SENDER_TRANS_OPTIONS
        ):
            error = _localize_error("signup.error.gender_trans_required")
            logger.warning(
                f"Signup failed: invalid gender/trans for final_sender {prolific_id} (gender={gender}, trans={trans})"
            )
        elif role_param == "final_sender" and User.query.filter_by(unique_id=prolific_id).first():
            # Final sender: existing user (returning sender from phase 1)
            existing_user = User.query.filter_by(unique_id=prolific_id).first()
            if existing_user.role != "sender":
                error = _localize_error("signup.error.account_exists")
                logger.warning(
                    f"Final sender signup failed: user {prolific_id} exists but is not a sender (role={existing_user.role})"
                )
            else:
                # Returning sender — check they have validated entries
                validated_entries = ConversationEntry.query.filter_by(
                    sender_id=existing_user.id,
                    status=ENTRY_STATUS_VALIDATED,
                ).all()

                if validated_entries:
                    # Happy path: assign them their own validated entries
                    try:
                        for entry in validated_entries:
                            entry.final_sender_id = existing_user.id
                        existing_user.mark_terms_accepted()
                        existing_user.instructions_read_at = None
                        db.session.commit()
                        logger.info(
                            f"Returning sender {prolific_id} signed up as final_sender on "
                            f"{len(validated_entries)} own entries"
                        )
                        login_user(existing_user)
                        session["login_nonce"] = existing_user.login_nonce
                        flash(
                            "Welcome back and thank you for joining the final part of our study.",
                            "success",
                        )
                        return redirect(get_post_login_redirect(existing_user))
                    except SQLAlchemyError as exc:
                        db.session.rollback()
                        logger.exception(
                            f"Database error during final sender re-signup for {prolific_id}: {exc}"
                        )
                        error = _localize_error("signup.error.generic")
                else:
                    # Edge case: all own entries are already completed —
                    # try to assign validated entries from the general pool.
                    all_completed = ConversationEntry.query.filter_by(
                        sender_id=existing_user.id,
                    ).filter(
                        ConversationEntry.status != ENTRY_STATUS_COMPLETED,
                    ).count() == 0

                    if not all_completed:
                        error = _localize_error("signup.error.not_enough_final_sender_entries")
                        logger.warning(
                            f"Final sender signup failed: returning sender {prolific_id} "
                            f"has no validated entries (and not all are completed)"
                        )
                    else:
                        logger.info(
                            f"Returning sender {prolific_id}: all own entries completed, "
                            f"falling back to general pool assignment"
                        )
                        try:
                            pool_error = _assign_final_sender_tasks(existing_user)
                            if pool_error:
                                error = _localize_error(pool_error)
                            else:
                                existing_user.mark_terms_accepted()
                                existing_user.instructions_read_at = None
                                db.session.commit()
                                logger.info(
                                    f"Returning sender {prolific_id} signed up as final_sender "
                                    f"via general pool assignment"
                                )
                                login_user(existing_user)
                                session["login_nonce"] = existing_user.login_nonce
                                flash(
                                    "Welcome back and thank you for joining the final part of our study.",
                                    "success",
                                )
                                return redirect(get_post_login_redirect(existing_user))
                        except SQLAlchemyError as exc:
                            db.session.rollback()
                            logger.exception(
                                f"Database error during final sender pool assignment for {prolific_id}: {exc}"
                            )
                            error = _localize_error("signup.error.generic")
                        error = _localize_error("signup.error.generic")
        elif User.query.filter_by(unique_id=prolific_id).first():
            error = _localize_error("signup.error.account_exists")
            logger.warning(f"Signup failed: account already exists for {prolific_id}")
        else:
            try:
                # Determine the stored role: final_sender users are stored as "sender"
                stored_role = "sender" if role_param == "final_sender" else role_param
                needs_gender = role_param in ("sender", "final_sender")
                logger.info(f"Creating new user: {prolific_id} (role: {stored_role})")
                user = User(
                    unique_id=prolific_id,
                    role=stored_role,
                    gender=gender if needs_gender else None,
                    trans=trans if needs_gender else None,
                )
                if needs_gender:
                    logger.info(
                        f"Sender {prolific_id} demographics: gender={gender}, trans={trans}"
                    )
                user.mark_terms_accepted()
                logger.info(f"User {prolific_id} accepted terms")
                db.session.add(user)
                db.session.flush()

                assignment_error = None
                if role_param == "sender":
                    assignment_error = _assign_sender_tasks(user)
                elif role_param == "final_sender":
                    assignment_error = _assign_final_sender_tasks(user)
                elif role_param == "receiver":
                    assignment_error = _assign_receiver_tasks(user)
                elif role_param == "validator":
                    assignment_error = _assign_validator_tasks(user)

                if assignment_error:
                    logger.error(
                        f"Task assignment failed for {prolific_id}: {assignment_error}"
                    )
                    raise ValueError(assignment_error)

                db.session.commit()
                logger.info(
                    f"User {prolific_id} successfully signed up and committed to database"
                )
            except ValueError as exc:
                db.session.rollback()
                error = _localize_error(str(exc))
            except SQLAlchemyError as exc:
                db.session.rollback()
                logger.exception(
                    f"Database error during signup for {prolific_id}: {exc}"
                )
                error = _localize_error("signup.error.generic")
            else:
                login_user(user)
                session["login_nonce"] = user.login_nonce
                logger.info(f"User {prolific_id} logged in after successful signup")
                if role_param == "final_sender":
                    flash("Welcome to the study.", "success")
                else:
                    flash(_localize_error("signup.success.welcome"), "success")
                return redirect(get_post_login_redirect(user))

    gender_options = [
        {"value": option, "label": SENDER_GENDER_LABELS.get(option, option.title())}
        for option in SENDER_GENDER_OPTIONS
    ]
    trans_options = [
        {
            "value": option,
            "label": SENDER_TRANS_LABELS.get(option, option.title()),
        }
        for option in SENDER_TRANS_OPTIONS
    ]

    template_user_role = role_param
    if (
        role_param == "final_sender"
        and authkey_param == ROLE_SIGNUP_KEYS.get("final_sender")
    ):
        template_user_role = "final_sender"

    return render_template(
        "signup.html",
        role=role_param,
        authkey=expected_key,
        error=error,
        prolific_id=prolific_id,
        gender=gender,
        trans=trans,
        gender_options=gender_options,
        trans_options=trans_options,
        user_role=template_user_role,
        terms_required=True,
        accepted_terms=accepted_terms,
        locale=locale,
    )


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(get_post_login_redirect(current_user))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(get_post_login_redirect(current_user))

    error = None

    if request.method == "POST":
        unique_id = request.form.get("unique_id", "").strip()
        password = request.form.get("password", "")

        if not unique_id:
            error = "Please provide your unique ID."
        else:
            user = User.query.filter_by(unique_id=unique_id).first()

            if not user:
                error = "Invalid credentials."
            elif user.requires_password and not user.check_password(password):
                error = "Invalid credentials."
            else:
                login_user(user)
                session["login_nonce"] = user.login_nonce
                flash("Welcome back!", "success")
                next_page = request.args.get("next")
                return redirect(next_page or get_post_login_redirect(user))

        if error is None:
            error = "Invalid credentials."

    return render_template("login.html", error=error)


@app.route("/maintenance")
def maintenance_page():
    if current_user.is_authenticated and current_user.role == "admin":
        return redirect(url_for("dashboard"))
    return render_template("maintenance.html")


def _detect_orphan_files() -> dict[str, Any]:
    """Detect orphan recordings and translations on disk that have no matching DB entry.

    Returns a dict with:
      - orphan_recordings: list of dicts with path, size, entry_id (extracted)
      - orphan_translations: list of dicts with path, size, entry_id (extracted)
      - total_orphan_size: combined size in bytes
    """
    import re

    recordings_dir = Path(app.config.get("RECORDINGS_DIR", "recordings"))
    translations_dir = Path(app.config.get("TRANSLATIONS_DIR", "translations"))

    # Get all existing entry IDs from the database
    existing_entry_ids: set[int] = set(
        row[0] for row in db.session.query(ConversationEntry.id).all()
    )

    orphan_recordings: list[dict[str, Any]] = []
    orphan_translations: list[dict[str, Any]] = []
    total_orphan_size = 0

    # Pattern to extract entry ID from recording paths
    # Recordings are stored as: entry_<id>/entry_<id>_<timestamp>.wav
    # or directly as: entry_<id>_<timestamp>.wav (demo files)
    recording_entry_pattern = re.compile(r"entry_(\d+)")

    # Scan recordings directory
    if recordings_dir.exists():
        for item in recordings_dir.iterdir():
            if item.is_dir():
                # Directory like entry_1/, entry_2/, etc.
                match = recording_entry_pattern.match(item.name)
                if match:
                    entry_id = int(match.group(1))
                    if entry_id not in existing_entry_ids:
                        # This entire folder is orphaned - sum up all files inside
                        folder_size = sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        )
                        file_count = sum(1 for f in item.rglob("*") if f.is_file())
                        orphan_recordings.append(
                            {
                                "path": str(item.relative_to(recordings_dir)),
                                "full_path": str(item),
                                "entry_id": entry_id,
                                "size": folder_size,
                                "file_count": file_count,
                                "is_folder": True,
                            }
                        )
                        total_orphan_size += folder_size
            elif item.is_file():
                # Direct file like demo_entry_2.wav or entry_1_timestamp.wav
                match = recording_entry_pattern.search(item.name)
                if match:
                    entry_id = int(match.group(1))
                    if entry_id not in existing_entry_ids:
                        file_size = item.stat().st_size
                        orphan_recordings.append(
                            {
                                "path": item.name,
                                "full_path": str(item),
                                "entry_id": entry_id,
                                "size": file_size,
                                "file_count": 1,
                                "is_folder": False,
                            }
                        )
                        total_orphan_size += file_size

    # Pattern to extract entry ID from translation files
    # Translations are stored as: entry_<id>.jsonl
    translation_entry_pattern = re.compile(r"entry_(\d+)\.jsonl$")

    # Scan translations directory
    if translations_dir.exists():
        for item in translations_dir.iterdir():
            if item.is_file():
                match = translation_entry_pattern.match(item.name)
                if match:
                    entry_id = int(match.group(1))
                    if entry_id not in existing_entry_ids:
                        file_size = item.stat().st_size
                        orphan_translations.append(
                            {
                                "path": item.name,
                                "full_path": str(item),
                                "entry_id": entry_id,
                                "size": file_size,
                            }
                        )
                        total_orphan_size += file_size

    # Sort by entry_id for consistent display
    orphan_recordings.sort(key=lambda x: x["entry_id"])
    orphan_translations.sort(key=lambda x: x["entry_id"])

    return {
        "orphan_recordings": orphan_recordings,
        "orphan_translations": orphan_translations,
        "total_orphan_size": total_orphan_size,
        "recordings_dir": str(recordings_dir),
        "translations_dir": str(translations_dir),
    }


def _format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if current_user.role != "admin":
        abort(403)

    conversation_upload_stats = None
    user_upload_stats = None
    conversation_entry_upload_stats = None
    receiver_assignment_stats = None
    validator_assignment_stats = None
    receiver_locale_options = list_receiver_locale_options()
    current_receiver_locale_code = _get_receiver_locale_code()
    current_receiver_locale = get_receiver_locale(current_receiver_locale_code)
    terms_enforcement_enabled = is_terms_enforcement_enabled()
    enforce_same_gender_enabled = is_enforce_same_gender_enabled()
    maintenance_enabled = is_maintenance_enabled()

    if request.method == "POST":
        form_name = request.form.get("form_name")

        if form_name == "conversation_upload":
            conversation_upload_stats = handle_conversation_upload(
                request.files.get("conversation_csv")
            )
        elif form_name == "user_upload":
            user_upload_stats = handle_user_upload(request.files.get("user_csv"))
        elif form_name == "conversation_entry_upload":
            conversation_entry_upload_stats = handle_conversation_entry_upload(
                request.files.get("conversation_entry_csv")
            )
        elif form_name == "receiver_assignment_upload":
            receiver_assignment_stats = handle_receiver_assignment_upload(
                request.files.get("receiver_assignment_csv")
            )
        elif form_name == "validator_assignment_upload":
            validator_assignment_stats = handle_validator_assignment_upload(
                request.files.get("validator_assignment_csv")
            )
        elif form_name == "receiver_locale":
            requested_code = (request.form.get("receiver_locale") or "").strip().lower()
            available_codes = {option["code"] for option in receiver_locale_options}
            if not requested_code or requested_code not in available_codes:
                logger.warning(
                    f"Admin {current_user.unique_id} attempted to set invalid receiver locale: {requested_code}"
                )
                flash("Invalid receiver interface language selected.", "warning")
            else:
                set_receiver_locale(requested_code)
                locale_obj = get_receiver_locale(requested_code)
                label = locale_obj.label if locale_obj else requested_code.upper()
                logger.info(
                    f"Admin {current_user.unique_id} changed receiver interface language to {label} ({requested_code})"
                )
                flash(
                    f"Receiver interface language updated to {label}.",
                    "success",
                )
            return redirect(url_for("dashboard"))
        elif form_name == "terms_enforcement":
            should_require = request.form.get("terms_required") == "1"
            set_terms_enforcement_enabled(should_require)
            logger.info(
                f"Admin {current_user.unique_id} {'enabled' if should_require else 'disabled'} terms enforcement"
            )
            message = (
                "Terms & Conditions requirement enabled."
                if should_require
                else "Terms & Conditions requirement disabled."
            )
            flash(message, "success")
            return redirect(url_for("dashboard"))
        elif form_name == "enforce_same_gender":
            enabled = request.form.get("enforce_same_gender") == "1"
            set_enforce_same_gender_enabled(enabled)
            logger.info(
                f"Admin {current_user.unique_id} {'enabled' if enabled else 'disabled'} enforce same gender"
            )
            flash(
                "Enforce same gender "
                + ("enabled." if enabled else "disabled."),
                "success",
            )
            return redirect(url_for("dashboard"))
        elif form_name == "maintenance_mode":
            enabled = request.form.get("maintenance_mode") == "1"
            set_maintenance_enabled(enabled)
            logger.info(
                "Admin %s %s maintenance mode",
                current_user.unique_id,
                "enabled" if enabled else "disabled",
            )
            flash(
                "Maintenance mode " + ("enabled." if enabled else "disabled."),
                "success",
            )
            return redirect(url_for("dashboard"))
        elif form_name == "increment_login_nonces":
            roles = request.form.getlist("nonce_roles")
            if not roles:
                roles = ["sender", "receiver", "validator"]
            valid_roles = [r for r in roles if r in VALID_USER_ROLES]
            if not valid_roles:
                flash("No valid roles selected.", "warning")
                return redirect(url_for("dashboard"))
            users = User.query.filter(User.role.in_(valid_roles)).all()
            count = 0
            for user in users:
                user.login_nonce += 1
                count += 1
            db.session.commit()
            logger.info(
                "Admin %s incremented login nonce for %d users (roles: %s)",
                current_user.unique_id,
                count,
                ", ".join(valid_roles),
            )
            flash(
                f"Incremented session nonce for {count} users across roles: {', '.join(valid_roles)}. "
                "All their existing sessions are now invalid.",
                "success",
            )
            return redirect(url_for("dashboard"))
        else:
            flash("Unknown action requested.", "warning")

    totals = {
        "conversations": Conversation.query.count(),
        "questions": Question.query.count(),
        "users": User.query.count(),
        "conversation_entries": ConversationEntry.query.count(),
    }

    role_distribution = [
        {
            "role": role,
            "label": label,
            "count": User.query.filter_by(role=role).count(),
        }
        for role, label in ROLE_DISPLAY_ORDER
    ]

    # Detect orphan files on disk
    orphan_data = _detect_orphan_files()
    # Add formatted sizes for display
    for item in orphan_data["orphan_recordings"]:
        item["size_formatted"] = _format_file_size(item["size"])
    for item in orphan_data["orphan_translations"]:
        item["size_formatted"] = _format_file_size(item["size"])
    orphan_data["total_orphan_size_formatted"] = _format_file_size(
        orphan_data["total_orphan_size"]
    )

    return render_template(
        "dashboard.html",
        totals=totals,
        role_distribution=role_distribution,
        conversation_upload_stats=conversation_upload_stats,
        user_upload_stats=user_upload_stats,
        conversation_entry_upload_stats=conversation_entry_upload_stats,
        receiver_assignment_stats=receiver_assignment_stats,
        validator_assignment_stats=validator_assignment_stats,
        receiver_locale_options=receiver_locale_options,
        current_receiver_locale=current_receiver_locale,
        current_receiver_locale_code=current_receiver_locale_code,
        terms_enforcement_enabled=terms_enforcement_enabled,
        enforce_same_gender_enabled=enforce_same_gender_enabled,
        maintenance_enabled=maintenance_enabled,
        orphan_data=orphan_data,
        signup_links={
            role: url_for("signup", role=role, authkey=key, _external=True)
            for role, key in ROLE_SIGNUP_KEYS.items()
        },
    )


@app.route("/admin/stats")
@login_required
def admin_stats():
    if current_user.role != "admin":
        abort(403)

    # Total conversation entries
    total_entries = ConversationEntry.query.count()

    # Counts by status
    status_rows = (
        db.session.query(ConversationEntry.status, func.count(ConversationEntry.id))
        .group_by(ConversationEntry.status)
        .all()
    )
    status_counts = {row[0]: row[1] for row in status_rows}

    # Sender assignments by gender
    sender_rows = (
        db.session.query(User.gender, func.count(ConversationEntry.id))
        .join(ConversationEntry, ConversationEntry.sender_id == User.id)
        .group_by(User.gender)
        .all()
    )
    raw_sender_counts = {row[0] or "unknown": row[1] for row in sender_rows}
    # Normalize to canonical sender genders
    sender_counts = {g: raw_sender_counts.get(g, 0) for g in SENDER_GENDER_OPTIONS}

    # For the dashboard we want counts grouped by sender gender
    # Count how many conversation entries have receivers/validators assigned, grouped by sender gender
    receiver_assigned_rows = (
        db.session.query(User.gender, func.count(ConversationEntry.id))
        .join(ConversationEntry, ConversationEntry.sender_id == User.id)
        .filter(ConversationEntry.receiver_id.isnot(None))
        .group_by(User.gender)
        .all()
    )
    raw_receiver_assigned_counts = {
        row[0] or "unknown": row[1] for row in receiver_assigned_rows
    }
    receiver_assigned_counts = {
        g: raw_receiver_assigned_counts.get(g, 0) for g in SENDER_GENDER_OPTIONS
    }

    validator_assigned_rows = (
        db.session.query(User.gender, func.count(ConversationEntry.id))
        .join(ConversationEntry, ConversationEntry.sender_id == User.id)
        .filter(ConversationEntry.validator_id.isnot(None))
        .group_by(User.gender)
        .all()
    )
    raw_validator_assigned_counts = {
        row[0] or "unknown": row[1] for row in validator_assigned_rows
    }
    validator_assigned_counts = {
        g: raw_validator_assigned_counts.get(g, 0) for g in SENDER_GENDER_OPTIONS
    }

    # Status counts by sender gender
    status_by_gender_rows = (
        db.session.query(
            User.gender, ConversationEntry.status, func.count(ConversationEntry.id)
        )
        .join(ConversationEntry, ConversationEntry.sender_id == User.id)
        .group_by(User.gender, ConversationEntry.status)
        .all()
    )

    # Build matrix: gender -> status -> count
    status_by_gender = {}
    for g, status, count in status_by_gender_rows:
        key = g or "unknown"
        status_by_gender.setdefault(key, {})[status] = count

    # Use canonical sender genders for reporting
    genders = list(SENDER_GENDER_OPTIONS)

    # Ensure status_by_gender has entries for each gender and each status
    for g in genders:
        status_by_gender.setdefault(g, {})
        for s in [
            ENTRY_STATUS_READY,
            ENTRY_STATUS_RECORDED,
            ENTRY_STATUS_TRANSLATED,
            ENTRY_STATUS_EVALUATED,
            ENTRY_STATUS_VALIDATED,
            ENTRY_STATUS_COMPLETED,
        ]:
            status_by_gender[g].setdefault(s, 0)

    # All statuses in canonical order
    statuses = [
        s
        for s in [
            ENTRY_STATUS_READY,
            ENTRY_STATUS_RECORDED,
            ENTRY_STATUS_TRANSLATED,
            ENTRY_STATUS_EVALUATED,
            ENTRY_STATUS_VALIDATED,
            ENTRY_STATUS_COMPLETED,
        ]
    ]

    return render_template(
        "admin_stats.html",
        total_entries=total_entries,
        status_counts=status_counts,
        sender_counts=sender_counts,
        receiver_assigned_counts=receiver_assigned_counts,
        validator_assigned_counts=validator_assigned_counts,
        status_by_gender=status_by_gender,
        genders=genders,
        statuses=statuses,
    )


@app.route("/conversations")
@login_required
def conversations():
    if current_user.role != "admin":
        abort(403)

    records = (
        Conversation.query.options(selectinload(Conversation.questions))
        .order_by(Conversation.created_at.desc())
        .all()
    )

    table_rows = [
        {
            "external_id": convo.external_id,
            "topic": convo.topic or "—",
            "source": convo.source or "—",
            "source_text": (convo.source_text or "").strip(),
            "question_count": len(convo.questions),
            "created_at": convo.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for convo in records
    ]

    return render_template("conversations.html", conversations=table_rows)


@app.route("/conversation-entries")
@login_required
def conversation_entries():
    if current_user.role != "admin":
        abort(403)

    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=100, type=int)
    allowed_per_page = {25, 50, 100, 200}
    if per_page not in allowed_per_page:
        per_page = 100

    pagination = (
        ConversationEntry.query.options(
            joinedload(ConversationEntry.conversation),
            joinedload(ConversationEntry.sender),
            joinedload(ConversationEntry.receiver),
            joinedload(ConversationEntry.validator),
            joinedload(ConversationEntry.final_sender),
            joinedload(ConversationEntry.final_survey),
        )
        .order_by(ConversationEntry.created_at.desc(), ConversationEntry.id.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    table_rows = []
    for entry in pagination.items:
        conversation = entry.conversation
        baseline_trust_score = (
            entry.final_survey.baseline_trust_score if entry.final_survey else None
        )
        baseline_reliance_score = (
            entry.final_survey.baseline_reliance_score if entry.final_survey else None
        )
        trust_score = entry.final_survey.trust_score if entry.final_survey else None
        reliance_score = (
            entry.final_survey.reliance_score if entry.final_survey else None
        )
        table_rows.append(
            {
                "id": entry.id,
                "conversation_id": conversation.external_id if conversation else "—",
                "conversation_topic": (
                    conversation.topic if conversation and conversation.topic else "—"
                ),
                "sender_id": entry.sender.unique_id if entry.sender else "—",
                "receiver_id": entry.receiver.unique_id if entry.receiver else "—",
                "validator_id": entry.validator.unique_id if entry.validator else "—",
                "final_sender_id": entry.final_sender.unique_id if entry.final_sender else "—",
                "status": entry.status,
                "status_label": ENTRY_STATUS_LABELS.get(
                    entry.status, entry.status.replace("_", " ").title()
                ),
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "survey_score": trust_score,
                "baseline_trust_score": baseline_trust_score,
                "baseline_reliance_score": baseline_reliance_score,
                "trust_score": trust_score,
                "reliance_score": reliance_score,
            }
        )

    return render_template(
        "conversation_entries.html",
        entries=table_rows,
        pagination=pagination,
        per_page=per_page,
        per_page_options=sorted(allowed_per_page),
    )


@app.route("/conversation-entries/export")
@login_required
def export_conversation_entries():
    if current_user.role != "admin":
        abort(403)

    entries = (
        ConversationEntry.query.options(
            joinedload(ConversationEntry.conversation),
            joinedload(ConversationEntry.sender),
            joinedload(ConversationEntry.receiver),
            joinedload(ConversationEntry.validator),
            joinedload(ConversationEntry.final_sender),
            joinedload(ConversationEntry.final_survey),
        )
        .order_by(ConversationEntry.created_at.asc())
        .all()
    )

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "conversation_entry_id",
            "conversation_id",
            "conversation_topic",
            "sender_id",
            "receiver_id",
            "validator_id",
            "final_sender_id",
            "sender_baseline_trust_score",
            "sender_baseline_reliance_score",
            "sender_trust_score",
            "sender_reliance_score",
            "status",
            "created_at",
        ]
    )

    for entry in entries:
        conversation = entry.conversation
        baseline_trust_score = (
            entry.final_survey.baseline_trust_score if entry.final_survey else None
        )
        baseline_reliance_score = (
            entry.final_survey.baseline_reliance_score if entry.final_survey else None
        )
        trust_score = entry.final_survey.trust_score if entry.final_survey else None
        reliance_score = (
            entry.final_survey.reliance_score if entry.final_survey else None
        )
        writer.writerow(
            [
                entry.id,
                conversation.external_id if conversation else "",
                conversation.topic if conversation and conversation.topic else "",
                entry.sender.unique_id if entry.sender else "",
                entry.receiver.unique_id if entry.receiver else "",
                entry.validator.unique_id if entry.validator else "",
                entry.final_sender.unique_id if entry.final_sender else "",
                baseline_trust_score if baseline_trust_score is not None else "",
                baseline_reliance_score if baseline_reliance_score is not None else "",
                trust_score if trust_score is not None else "",
                reliance_score if reliance_score is not None else "",
                entry.status,
                entry.created_at.strftime("%Y-%m-%d %H:%M"),
            ]
        )

    csv_content = output.getvalue()
    output.close()
    filename = f"conversation_entries_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@app.route("/conversation-entries/bulk-delete", methods=["POST"])
@login_required
def bulk_delete_entries():
    """Bulk-delete selected conversation entries and all associated artifacts."""
    if current_user.role != "admin":
        abort(403)

    data = request.get_json(silent=True)
    if not data or not isinstance(data.get("entry_ids"), list):
        return jsonify(success=False, error="Missing or invalid entry_ids."), 400

    entry_ids = [int(eid) for eid in data["entry_ids"] if str(eid).isdigit()]
    if not entry_ids:
        return jsonify(success=False, error="No valid entry IDs provided."), 400

    entries = ConversationEntry.query.filter(
        ConversationEntry.id.in_(entry_ids)
    ).all()
    if not entries:
        return jsonify(success=False, error="No matching entries found."), 404

    deleted_count = 0
    try:
        for entry in entries:
            # Delete final survey
            if entry.final_survey is not None:
                db.session.delete(entry.final_survey)
            # Delete validator review
            if hasattr(entry, "validator_review") and entry.validator_review is not None:
                db.session.delete(entry.validator_review)
            # Delete validator evaluations
            for evaluation in list(entry.validator_evaluations or []):
                db.session.delete(evaluation)
            # Delete receiver responses
            for response in list(entry.receiver_responses or []):
                db.session.delete(response)
            # Delete recording from disk and DB
            if entry.recording is not None:
                _delete_entry_recording_from_disk(entry.recording)
                db.session.delete(entry.recording)
            db.session.delete(entry)
            deleted_count += 1

        db.session.commit()
        logger.info(
            "Admin %s bulk-deleted %d conversation entries: %s",
            current_user.unique_id,
            deleted_count,
            entry_ids,
        )
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception("Bulk delete failed: %s", exc)
        return jsonify(success=False, error="Database error during deletion."), 500

    return jsonify(success=True, deleted=deleted_count)


@app.route("/conversation-entries/bulk-update-role", methods=["POST"])
@login_required
def bulk_update_entry_role():
    """Bulk-update a role assignment (sender, receiver, validator, final_sender) on selected entries."""
    if current_user.role != "admin":
        abort(403)

    data = request.get_json(silent=True)
    if not data or not isinstance(data.get("entry_ids"), list):
        return jsonify(success=False, error="Missing or invalid entry_ids."), 400

    entry_ids = [int(eid) for eid in data["entry_ids"] if str(eid).isdigit()]
    if not entry_ids:
        return jsonify(success=False, error="No valid entry IDs provided."), 400

    role_field = data.get("role_field")
    allowed_fields = {"sender_id", "receiver_id", "validator_id", "final_sender_id"}
    if role_field not in allowed_fields:
        return jsonify(
            success=False,
            error=f"Invalid role_field. Must be one of: {', '.join(sorted(allowed_fields))}.",
        ), 400

    new_unique_id = (data.get("new_unique_id") or "").strip()
    # Empty string means "remove assignment"
    new_user_id = None
    if new_unique_id:
        target_user = User.query.filter_by(unique_id=new_unique_id).first()
        if not target_user:
            return jsonify(
                success=False,
                error=f"User '{new_unique_id}' not found.",
            ), 404
        new_user_id = target_user.id

    entries = ConversationEntry.query.filter(
        ConversationEntry.id.in_(entry_ids)
    ).all()
    if not entries:
        return jsonify(success=False, error="No matching entries found."), 404

    updated_count = 0
    try:
        for entry in entries:
            setattr(entry, role_field, new_user_id)
            updated_count += 1
        db.session.commit()
        action = f"set to {new_unique_id}" if new_unique_id else "cleared"
        logger.info(
            "Admin %s bulk-updated %s on %d entries (%s): %s",
            current_user.unique_id,
            role_field,
            updated_count,
            action,
            entry_ids,
        )
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception("Bulk role update failed: %s", exc)
        return jsonify(success=False, error="Database error during update."), 500

    return jsonify(success=True, updated=updated_count)


@app.route("/admin/export-snapshot.json")
@login_required
def export_snapshot_json():
    """Export the full snapshot as a streaming JSON array."""
    if current_user.role != "admin":
        abort(403)

    logger.info(f"Admin {current_user.unique_id} requested JSON snapshot export")

    recordings_dir = Path(app.config.get("RECORDINGS_DIR", "recordings"))
    filename = f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    def generate_json():
        yield "["
        first = True

        # Use selectinload for collections; joinedload for scalars.
        query = (
            ConversationEntry.query.options(
                joinedload(ConversationEntry.conversation).selectinload(
                    Conversation.questions
                ),
                joinedload(ConversationEntry.sender),
                joinedload(ConversationEntry.receiver),
                joinedload(ConversationEntry.validator),
                joinedload(ConversationEntry.recording),
                selectinload(ConversationEntry.receiver_responses).joinedload(
                    ReceiverResponse.question
                ),
                joinedload(ConversationEntry.validator_review),
                selectinload(ConversationEntry.validator_evaluations).joinedload(
                    ValidatorResponseEvaluation.question
                ),
                joinedload(ConversationEntry.final_survey),
            )
            .order_by(ConversationEntry.id.asc())
            .yield_per(100)
        )

        count = 0
        for entry in query:
            conv = entry.conversation
            questions_list = []
            if conv and conv.questions:
                for q in sorted(conv.questions, key=lambda x: x.id):
                    questions_list.append(
                        {
                            "question_id": q.question_id,
                            "question_text": q.question_text,
                            "answer": q.answer,
                            "translated_question": q.translated_question,
                        }
                    )

            receiver_responses_list = []
            if entry.receiver_responses:
                for resp in sorted(entry.receiver_responses, key=lambda x: x.question_id):
                    question = resp.question
                    receiver_responses_list.append(
                        {
                            "question_id": question.question_id
                            if question
                            else str(resp.question_id),
                            "response_text": resp.response_text,
                        }
                    )

            validator_evaluations_list = []
            if entry.validator_evaluations:
                for ev in sorted(entry.validator_evaluations, key=lambda x: x.question_id):
                    question = ev.question
                    validator_evaluations_list.append(
                        {
                            "question_id": question.question_id
                            if question
                            else str(ev.question_id),
                            "is_correct": ev.is_correct,
                        }
                    )

            recording_path = None
            if entry.recording:
                recording_path = str(recordings_dir / entry.recording.stored_filename)

            validator_translation_score = None
            if entry.validator_review:
                validator_translation_score = entry.validator_review.translation_score

            validator_corrections = 0
            validator_question_total = len(questions_list)
            if entry.validator_evaluations:
                validator_corrections = sum(
                    1 for ev in entry.validator_evaluations if not ev.is_correct
                )

            final_survey = entry.final_survey
            final_survey_data: dict[str, Any] = {}
            if final_survey:
                final_survey_data = {
                    "baseline_usefulness_score": final_survey.baseline_usefulness_score,
                    "baseline_satisfaction_score": final_survey.baseline_satisfaction_score,
                    "baseline_trust_score": final_survey.baseline_trust_score,
                    "baseline_reliance_score": final_survey.baseline_reliance_score,
                    "usefulness_score": final_survey.usefulness_score,
                    "satisfaction_score": final_survey.satisfaction_score,
                    "trust_score": final_survey.trust_score,
                    "reliance_score": final_survey.reliance_score,
                    "submitted_at": final_survey.updated_at.strftime("%Y-%m-%d %H:%M"),
                }

            record = {
                "entry_id": entry.id,
                "conversation_id": conv.external_id if conv else None,
                "conversation_topic": conv.topic if conv else None,
                "conversation_source": conv.source if conv else None,
                "conversation_text": conv.source_text if conv else None,
                "status": entry.status,
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_unique_id": entry.sender.unique_id if entry.sender else None,
                "sender_gender": entry.sender.gender if entry.sender else None,
                "receiver_unique_id": entry.receiver.unique_id if entry.receiver else None,
                "validator_unique_id": entry.validator.unique_id if entry.validator else None,
                "final_sender_unique_id": entry.final_sender.unique_id if entry.final_sender else None,
                "recording_path": recording_path,
                "translation_text": entry.translation_text or None,
                "translation_model": getattr(entry, "translation_model", None),
                "questions": questions_list,
                "receiver_responses": receiver_responses_list,
                "validator_translation_score": validator_translation_score,
                "validator_evaluations": validator_evaluations_list,
                "validator_corrections": validator_corrections,
                "validator_question_total": validator_question_total,
                "final_survey": final_survey_data,
            }

            prefix = "" if first else ","
            first = False
            yield prefix + json.dumps(record, ensure_ascii=False)
            count += 1

        yield "]"
        logger.info(f"JSON snapshot export completed: {count} entries streamed to {filename}")

    return Response(
        stream_with_context(generate_json()),
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@app.route("/users")
@login_required
def users():
    if current_user.role != "admin":
        abort(403)

    user_records = User.query.order_by(User.created_at.desc()).all()
    user_ids = [user.id for user in user_records]
    user_task_stats = _get_user_task_stats(user_ids)

    user_rows: list[dict[str, Any]] = []
    for user in user_records:
        row = _user_to_dict(user, include_recordings=False)
        stats = user_task_stats.get(user.id, {})

        if user.role == "sender":
            row["tasks_total"] = stats.get("tasks_total", 0)
            row["tasks_sender_validated_or_below"] = stats.get(
                "tasks_sender_validated_or_below", 0
            )
            row["tasks_sender_completed"] = stats.get("tasks_sender_completed", 0)
        elif user.role == "receiver":
            row["tasks_total"] = stats.get("tasks_total", 0)
            row["tasks_receiver_evaluated"] = stats.get("tasks_receiver_evaluated", 0)
        elif user.role == "validator":
            row["tasks_total"] = stats.get("tasks_total", 0)
            row["tasks_validator_validated"] = stats.get("tasks_validator_validated", 0)
        else:
            row["tasks_total"] = 0

        user_rows.append(row)

    gender_options = [
        {"value": option, "label": SENDER_GENDER_LABELS.get(option, option.title())}
        for option in SENDER_GENDER_OPTIONS
    ]
    trans_options = [
        {"value": option, "label": SENDER_TRANS_LABELS.get(option, option.title())}
        for option in SENDER_TRANS_OPTIONS
    ]

    return render_template(
        "users.html",
        users=user_rows,
        role_options=ROLE_DISPLAY_ORDER,
        gender_options=gender_options,
        trans_options=trans_options,
    )


@app.route("/users/<int:user_id>/entries")
@login_required
def user_entries(user_id: int):
    if current_user.role != "admin":
        abort(403)

    user = _session_get_or_404(User, user_id)

    entries = (
        ConversationEntry.query.options(
            joinedload(ConversationEntry.conversation),
            joinedload(ConversationEntry.recording),
            selectinload(ConversationEntry.receiver_responses),
        )
        .filter(
            or_(
                ConversationEntry.sender_id == user.id,
                ConversationEntry.receiver_id == user.id,
                ConversationEntry.validator_id == user.id,
                ConversationEntry.final_sender_id == user.id,
            )
        )
        .order_by(ConversationEntry.created_at.desc())
        .all()
    )

    rows = []
    tasks_completed_count = 0
    for entry in entries:
        if entry.final_sender_id == user.id:
            participant_role = "Final Sender"
            # Final sender task is completed when status is 'completed'
            is_task_completed = entry.status == ENTRY_STATUS_COMPLETED
        elif entry.sender_id == user.id:
            participant_role = "Sender"
            # Sender task is completed when status is 'completed'
            is_task_completed = entry.status == ENTRY_STATUS_COMPLETED
        elif entry.receiver_id == user.id:
            participant_role = "Receiver"
            # Receiver task is completed when status is 'evaluated'
            is_task_completed = entry.status == ENTRY_STATUS_EVALUATED
        else:
            participant_role = "Validator"
            # Validator task is completed when status is 'validated' or 'completed'
            is_task_completed = entry.status in {
                ENTRY_STATUS_VALIDATED,
                ENTRY_STATUS_COMPLETED,
            }

        if is_task_completed:
            tasks_completed_count += 1

        recording = entry.recording

        latest_activity = entry.created_at
        if (
            recording
            and recording.created_at
            and recording.created_at > latest_activity
        ):
            latest_activity = recording.created_at
        if entry.receiver_responses:
            response_latest = max(
                (response.updated_at or response.created_at)
                for response in entry.receiver_responses
            )
            if response_latest > latest_activity:
                latest_activity = response_latest

        rows.append(
            {
                "entry_id": entry.id,
                "conversation_id": (
                    entry.conversation.external_id if entry.conversation else "—"
                ),
                "conversation_topic": (
                    entry.conversation.topic
                    if entry.conversation and entry.conversation.topic
                    else "—"
                ),
                "status": entry.status,
                "status_label": ENTRY_STATUS_LABELS.get(
                    entry.status, entry.status.replace("_", " ").title()
                ),
                "participant_role": participant_role,
                "has_recording": bool(recording),
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "last_activity": latest_activity.strftime("%Y-%m-%d %H:%M"),
                "detail_url": url_for(
                    "user_entry_detail", user_id=user.id, entry_id=entry.id
                ),
                "can_downgrade": entry.status != ENTRY_STATUS_READY,
                "downgrade_options": [
                    {
                        "value": status,
                        "label": ENTRY_STATUS_LABELS.get(status, status.title()),
                    }
                    for status in ENTRY_STATUS_ORDER
                    if _get_status_index(status) < _get_status_index(entry.status)
                ],
            }
        )

    user_summary = {
        "id": user.id,
        "unique_id": user.unique_id,
        "role_label": ROLE_LABELS.get(user.role, user.role.title()),
        "entries_total": len(rows),
        "tasks_completed": tasks_completed_count,
    }

    return render_template(
        "user_entries.html",
        user_summary=user_summary,
        entries=rows,
        entry_status_order=ENTRY_STATUS_ORDER,
    )


@app.route("/users/<int:user_id>/entries/<int:entry_id>/downgrade", methods=["POST"])
@login_required
def admin_downgrade_entry(user_id: int, entry_id: int):
    if current_user.role != "admin":
        abort(403)

    _session_get_or_404(User, user_id)
    entry = _session_get_or_404(ConversationEntry, entry_id)

    current_status = entry.status
    if current_status == ENTRY_STATUS_READY:
        return abort(400, description="Ready entries cannot be downgraded.")

    target_status = (request.form.get("target_status") or "").strip()
    if target_status not in _ENTRY_STATUS_INDEX:
        return abort(400, description="Unknown target status.")

    current_idx = _get_status_index(current_status)
    target_idx = _get_status_index(target_status)
    if target_idx >= current_idx:
        return abort(
            400, description="Target status must be earlier than current status."
        )

    impacts = _entry_downgrade_impacts(current_status, target_status)
    logger.info(
        "Admin %s downgrading entry %s from %s -> %s (impacts=%s)",
        current_user.unique_id,
        entry.id,
        current_status,
        target_status,
        impacts,
    )

    try:
        # Remove artifacts for any states strictly after the target state.
        if current_idx >= _get_status_index(
            ENTRY_STATUS_COMPLETED
        ) and target_idx < _get_status_index(ENTRY_STATUS_COMPLETED):
            if entry.final_survey is not None:
                db.session.delete(entry.final_survey)

        if current_idx >= _get_status_index(
            ENTRY_STATUS_VALIDATED
        ) and target_idx < _get_status_index(ENTRY_STATUS_VALIDATED):
            if entry.validator_review is not None:
                db.session.delete(entry.validator_review)
            for evaluation in list(entry.validator_evaluations or []):
                db.session.delete(evaluation)

        if current_idx >= _get_status_index(
            ENTRY_STATUS_EVALUATED
        ) and target_idx < _get_status_index(ENTRY_STATUS_EVALUATED):
            for response in list(entry.receiver_responses or []):
                db.session.delete(response)

        if current_idx >= _get_status_index(
            ENTRY_STATUS_TRANSLATED
        ) and target_idx < _get_status_index(ENTRY_STATUS_TRANSLATED):
            entry.translation_text = None
            entry.translation_model = None

        if current_idx >= _get_status_index(
            ENTRY_STATUS_RECORDED
        ) and target_idx < _get_status_index(ENTRY_STATUS_RECORDED):
            if entry.recording is not None:
                _delete_entry_recording_from_disk(entry.recording)
                db.session.delete(entry.recording)

        entry.status = target_status
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception("Failed to downgrade conversation entry", exc_info=exc)
        flash("Failed to downgrade entry.", "danger")
        return redirect(url_for("user_entries", user_id=user_id))

    if impacts:
        flash(
            f"Entry {entry.id} downgraded to {ENTRY_STATUS_LABELS.get(target_status, target_status)}. "
            + "Removed: "
            + ", ".join(impacts)
            + ".",
            "warning",
        )
    else:
        flash(
            f"Entry {entry.id} downgraded to {ENTRY_STATUS_LABELS.get(target_status, target_status)}.",
            "warning",
        )

    return redirect(url_for("user_entries", user_id=user_id))


@app.route("/users/<int:user_id>/entries/<int:entry_id>")
@login_required
def user_entry_detail(user_id: int, entry_id: int):
    if current_user.role != "admin":
        abort(403)

    user = _session_get_or_404(User, user_id)
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if user.id not in {
        entry.sender_id,
        entry.receiver_id,
        entry.validator_id,
        entry.final_sender_id,
    }:
        abort(404)

    if entry.final_sender_id == user.id:
        participant_role = "Final Sender"
    elif entry.sender_id == user.id:
        participant_role = "Sender"
    elif entry.receiver_id == user.id:
        participant_role = "Receiver"
    else:
        participant_role = "Validator"

    return _render_conversation_entry_detail(
        entry=entry,
        participant_role=participant_role,
        back_url=url_for("user_entries", user_id=user.id),
        back_label=f"Back to {user.unique_id}'s entries",
    )


def _render_conversation_entry_detail(
    *,
    entry: ConversationEntry,
    participant_role: str,
    back_url: str,
    back_label: str,
):
    """Render the admin entry detail page for an entry."""

    recording = entry.recording
    recording_summary: Optional[dict[str, Any]] = None
    if recording:
        recording_summary = {
            "url": url_for("admin_entry_recording", entry_id=entry.id),
            "filename": recording.original_filename
            or Path(recording.stored_filename).name,
            "created_at": recording.created_at.strftime("%Y-%m-%d %H:%M"),
            "mimetype": recording.file_mimetype or "audio/wav",
            "file_size": recording.file_size,
        }

    responses_by_question_id = {
        response.question_id: (response.response_text or "").strip()
        for response in entry.receiver_responses
    }
    validator_checks = {
        evaluation.question_id: evaluation.is_correct
        for evaluation in entry.validator_evaluations
    }

    questions = (
        Question.query.filter_by(conversation_id=entry.conversation_id)
        .order_by(Question.id.asc())
        .all()
    )

    question_rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        question_rows.append(
            {
                "id": question.id,
                "number": index,
                "question_text": (question.question_text or "").strip(),
                "translated_question": (
                    question.translated_question or question.question_text or ""
                ).strip(),
                "reference_answer": (question.answer or "").strip(),
                "receiver_response": responses_by_question_id.get(question.id, ""),
                "validator_correct": validator_checks.get(question.id),
            }
        )

    latest_activity = entry.created_at
    if recording and recording.created_at and recording.created_at > latest_activity:
        latest_activity = recording.created_at
    if entry.receiver_responses:
        response_latest = max(
            (response.updated_at or response.created_at)
            for response in entry.receiver_responses
        )
        if response_latest > latest_activity:
            latest_activity = response_latest

    entry_summary = {
        "entry_id": entry.id,
        "conversation_id": (
            entry.conversation.external_id if entry.conversation else "—"
        ),
        "conversation_topic": (
            entry.conversation.topic
            if entry.conversation and entry.conversation.topic
            else "—"
        ),
        "status": entry.status,
        "status_label": entry.status.replace("_", " ").title(),
        "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M"),
        "last_activity": latest_activity.strftime("%Y-%m-%d %H:%M"),
        "participant_role": participant_role,
    }

    participants = {
        "sender": {
            "label": "Sender",
            "id": entry.sender.id if entry.sender else None,
            "unique_id": entry.sender.unique_id if entry.sender else "—",
            "role_label": (
                ROLE_LABELS.get(entry.sender.role, entry.sender.role.title())
                if entry.sender
                else "Sender"
            ),
        },
        "receiver": {
            "label": "Receiver",
            "id": entry.receiver.id if entry.receiver else None,
            "unique_id": entry.receiver.unique_id if entry.receiver else "—",
            "role_label": (
                ROLE_LABELS.get(entry.receiver.role, entry.receiver.role.title())
                if entry.receiver
                else "Receiver"
            ),
        },
        "validator": {
            "label": "Validator",
            "id": entry.validator.id if entry.validator else None,
            "unique_id": entry.validator.unique_id if entry.validator else "—",
            "role_label": (
                ROLE_LABELS.get(entry.validator.role, entry.validator.role.title())
                if entry.validator
                else "Validator"
            ),
        },
        "final_sender": {
            "label": "Final Sender",
            "id": entry.final_sender.id if entry.final_sender else None,
            "unique_id": entry.final_sender.unique_id if entry.final_sender else "—",
            "role_label": (
                ROLE_LABELS.get(entry.final_sender.role, entry.final_sender.role.title())
                if entry.final_sender
                else "Final Sender"
            ),
        },
    }

    conversation_text = entry.conversation.source_text if entry.conversation else ""
    translation_text = (entry.translation_text or "").strip()
    translation_model = getattr(entry, "translation_model", None)
    validator_review = entry.validator_review
    validator_score = validator_review.translation_score if validator_review else None

    # Count how many questions were marked incorrect (i.e., needing correction) by the validator
    validator_corrected_count = sum(
        1 for q in question_rows if q.get("validator_correct") is False
    )
    validator_question_total = len(question_rows)

    final_survey = entry.final_survey
    final_survey_summary: Optional[dict[str, Any]] = None
    if final_survey:
        final_survey_summary = {
            "baseline_satisfaction_score": final_survey.baseline_satisfaction_score,
            "baseline_trust_score": final_survey.baseline_trust_score,
            "baseline_reliance_score": final_survey.baseline_reliance_score,
            "satisfaction_score": final_survey.satisfaction_score,
            "trust_score": final_survey.trust_score,
            "reliance_score": final_survey.reliance_score,
            "submitted_at": final_survey.updated_at.strftime("%Y-%m-%d %H:%M"),
        }

    return render_template(
        "conversation_entry_detail.html",
        back_url=back_url,
        back_label=back_label,
        entry_summary=entry_summary,
        recording=recording_summary,
        participants=participants,
        question_rows=question_rows,
        conversation_text=(conversation_text or "").strip(),
        translation_text=translation_text,
        translation_model=translation_model,
        validator_score=validator_score,
        validator_review=validator_review,
        validator_corrected_count=validator_corrected_count,
        validator_question_total=validator_question_total,
        final_survey=final_survey_summary,
    )


@app.route("/conversation-entries/<int:entry_id>")
@login_required
def conversation_entry_detail(entry_id: int):
    if current_user.role != "admin":
        abort(403)

    entry = _session_get_or_404(ConversationEntry, entry_id)

    return _render_conversation_entry_detail(
        entry=entry,
        participant_role="Entry Overview",
        back_url=url_for("conversation_entries"),
        back_label="Back to conversation entries",
    )


@app.route("/admin/entries/<int:entry_id>/recording")
@login_required
def admin_entry_recording(entry_id: int):
    if current_user.role != "admin":
        abort(403)

    entry = _session_get_or_404(ConversationEntry, entry_id)
    recording = entry.recording

    if not recording:
        abort(404)

    recordings_dir = _ensure_recordings_dir().resolve()
    file_path = (recordings_dir / recording.stored_filename).resolve()

    try:
        file_path.relative_to(recordings_dir)
    except ValueError:
        abort(404)

    if not file_path.exists():
        abort(404)

    return send_file(
        file_path,
        mimetype=recording.file_mimetype or "audio/wav",
        download_name=recording.original_filename or file_path.name,
        as_attachment=False,
    )


@app.route("/users/create", methods=["POST"])
@login_required
def create_user():
    if current_user.role != "admin":
        abort(403)

    payload = request.get_json(silent=True) or {}
    unique_id = _clean_value(payload.get("unique_id"))
    role = _normalize_role(payload.get("role"))
    password = _clean_password(payload.get("password"))
    gender = _clean_value(payload.get("gender"))
    trans = _clean_value(payload.get("trans"))

    if not unique_id:
        return jsonify(success=False, error="Unique ID is required."), 400
    if not role:
        return (
            jsonify(
                success=False,
                error="Role must be sender, receiver, validator, or admin.",
            ),
            400,
        )

    if User.query.filter_by(unique_id=unique_id).first():
        return (
            jsonify(success=False, error="A user with that unique ID already exists."),
            400,
        )

    if gender is not None and gender not in SENDER_GENDER_OPTIONS:
        return (
            jsonify(success=False, error="Invalid gender value."),
            400,
        )
    if trans is not None and trans not in SENDER_TRANS_OPTIONS:
        return (
            jsonify(success=False, error="Invalid trans value."),
            400,
        )

    user = User(unique_id=unique_id, role=role, gender=gender, trans=trans)
    if role == "admin":
        user.terms_accepted_at = datetime.now(timezone.utc)
        if password:
            user.set_password(password)
        else:
            return (
                jsonify(success=False, error="Admin users must have a password."),
                400,
            )
    elif password:
        user.set_password(password)

    db.session.add(user)

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception("Failed to create user", exc_info=exc)
        return jsonify(success=False, error="Database error while creating user."), 500

    return jsonify(success=True, user=_user_to_dict(user))


@app.route("/users/<int:user_id>/update", methods=["POST"])
@login_required
def update_user(user_id: int):
    if current_user.role != "admin":
        abort(403)

    user = _session_get_or_404(User, user_id)
    payload = request.get_json(silent=True) or {}

    new_unique_id = _clean_value(payload.get("unique_id"))
    new_role = _normalize_role(payload.get("role"))
    password = _clean_password(payload.get("password"))
    gender = _clean_value(payload.get("gender"))
    trans = _clean_value(payload.get("trans"))

    if not new_unique_id:
        return jsonify(success=False, error="Unique ID is required."), 400
    if not new_role:
        return (
            jsonify(
                success=False,
                error="Role must be sender, receiver, validator, or admin.",
            ),
            400,
        )

    if user.id == current_user.id and new_role != "admin":
        return (
            jsonify(
                success=False, error="You must remain an admin to manage the platform."
            ),
            400,
        )

    if (
        user.unique_id != new_unique_id
        and User.query.filter_by(unique_id=new_unique_id).first()
    ):
        return (
            jsonify(success=False, error="A user with that unique ID already exists."),
            400,
        )

    if user.role == "admin" and new_role != "admin":
        remaining_admins = User.query.filter_by(role="admin").count()
        if remaining_admins <= 1:
            return (
                jsonify(
                    success=False,
                    error="At least one admin must remain on the platform.",
                ),
                400,
            )

    previous_role = user.role
    user.unique_id = new_unique_id
    user.role = new_role

    if gender is not None and gender not in SENDER_GENDER_OPTIONS:
        return (
            jsonify(success=False, error="Invalid gender value."),
            400,
        )
    if trans is not None and trans not in SENDER_TRANS_OPTIONS:
        return (
            jsonify(success=False, error="Invalid trans value."),
            400,
        )

    user.gender = gender
    user.trans = trans

    if previous_role == "admin" and new_role != "admin":
        user.terms_accepted_at = None
    elif new_role == "admin" and not user.terms_accepted_at:
        user.terms_accepted_at = datetime.now(timezone.utc)

    if new_role == "admin":
        if password:
            user.set_password(password)
        elif not user.password_hash:
            return (
                jsonify(success=False, error="Admin users must have a password."),
                400,
            )
    else:
        if password:
            user.set_password(password)
        else:
            user.password_hash = None

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception("Failed to update user", exc_info=exc)
        return jsonify(success=False, error="Database error while updating user."), 500

    return jsonify(success=True, user=_user_to_dict(user))


@app.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
def delete_user(user_id: int):
    if current_user.role != "admin":
        abort(403)

    user = _session_get_or_404(User, user_id)
    logger.info(
        f"Admin {current_user.unique_id} attempting to delete user {user.unique_id}"
    )

    if user.id == current_user.id:
        return jsonify(success=False, error="You cannot delete your own account."), 400

    if user.role == "admin":
        remaining_admins = User.query.filter_by(role="admin").count()
        if remaining_admins <= 1:
            return (
                jsonify(
                    success=False,
                    error="At least one admin must remain on the platform.",
                ),
                400,
            )

    # For senders, collect entry statistics before deletion for logging
    entry_deletion_stats: Optional[dict[str, Any]] = None
    if user.role == "sender":
        entries = ConversationEntry.query.filter_by(sender_id=user.id).all()
        total_entries = len(entries)
        status_counts: dict[str, int] = {}
        non_ready_count = 0

        for entry in entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
            if entry.status != ENTRY_STATUS_READY:
                non_ready_count += 1

        entry_deletion_stats = {
            "total": total_entries,
            "non_ready": non_ready_count,
            "by_status": status_counts,
        }

        # Also clear final_sender_id on any entries where this user is the final sender
        final_sender_entries = ConversationEntry.query.filter_by(
            final_sender_id=user.id,
        ).all()
        for entry in final_sender_entries:
            entry.final_sender_id = None
            # If the entry was completed via this user's survey, roll back
            if entry.status == ENTRY_STATUS_COMPLETED:
                if entry.final_survey:
                    db.session.delete(entry.final_survey)
                entry.status = ENTRY_STATUS_VALIDATED

    # For receivers: delete their responses and roll back entry statuses
    receiver_deletion_stats: Optional[dict[str, Any]] = None
    if user.role == "receiver":
        # Find all entries assigned to this receiver
        receiver_entries = ConversationEntry.query.filter_by(receiver_id=user.id).all()

        # Count responses that will be deleted
        responses_deleted = 0
        entries_rolled_back = 0

        for entry in receiver_entries:
            # Delete all ReceiverResponse instances for this entry
            response_count = ReceiverResponse.query.filter_by(
                conversation_entry_id=entry.id
            ).delete()
            responses_deleted += response_count

            # Roll back status from 'evaluated' to 'translated'
            if entry.status == ENTRY_STATUS_EVALUATED:
                entry.status = ENTRY_STATUS_TRANSLATED
                entries_rolled_back += 1

            # Clear the receiver assignment
            entry.receiver_id = None

        receiver_deletion_stats = {
            "total_entries": len(receiver_entries),
            "responses_deleted": responses_deleted,
            "entries_rolled_back": entries_rolled_back,
        }

    # For validators: delete their evaluations/reviews and roll back entry statuses
    validator_deletion_stats: Optional[dict[str, Any]] = None
    if user.role == "validator":
        # Find all entries assigned to this validator
        validator_entries = ConversationEntry.query.filter_by(
            validator_id=user.id
        ).all()

        # Count evaluations that will be deleted
        evaluations_deleted = 0
        reviews_deleted = 0
        entries_rolled_back = 0

        for entry in validator_entries:
            # Delete all ValidatorResponseEvaluation instances for this entry
            eval_count = ValidatorResponseEvaluation.query.filter_by(
                conversation_entry_id=entry.id
            ).delete()
            evaluations_deleted += eval_count

            # Delete ValidatorReview if exists
            review_count = ValidatorReview.query.filter_by(
                conversation_entry_id=entry.id
            ).delete()
            reviews_deleted += review_count

            # Roll back status from 'validated' to 'evaluated'
            if entry.status == ENTRY_STATUS_VALIDATED:
                entry.status = ENTRY_STATUS_EVALUATED
                entries_rolled_back += 1

            # Clear the validator assignment
            entry.validator_id = None

        validator_deletion_stats = {
            "total_entries": len(validator_entries),
            "evaluations_deleted": evaluations_deleted,
            "reviews_deleted": reviews_deleted,
            "entries_rolled_back": entries_rolled_back,
        }

    try:
        db.session.delete(user)
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(f"Failed to delete user {user.unique_id}: {exc}")
        return jsonify(success=False, error="Database error while deleting user."), 500

    # Log detailed info for sender deletions
    if entry_deletion_stats:
        logger.info(
            f"Deleted sender {user.unique_id}: {entry_deletion_stats['total']} conversation entries removed "
            f"({entry_deletion_stats['non_ready']} were not in 'ready' state). "
            f"Status breakdown: {entry_deletion_stats['by_status']}"
        )
    elif receiver_deletion_stats:
        logger.info(
            f"Deleted receiver {user.unique_id}: {receiver_deletion_stats['responses_deleted']} responses deleted, "
            f"{receiver_deletion_stats['entries_rolled_back']} entries rolled back to 'translated' status."
        )
    elif validator_deletion_stats:
        logger.info(
            f"Deleted validator {user.unique_id}: {validator_deletion_stats['evaluations_deleted']} evaluations deleted, "
            f"{validator_deletion_stats['reviews_deleted']} reviews deleted, "
            f"{validator_deletion_stats['entries_rolled_back']} entries rolled back to 'evaluated' status."
        )
    else:
        logger.info(f"Deleted user {user.unique_id} (role: {user.role})")

    return jsonify(success=True)


@app.route("/users/<int:user_id>/deletion-info")
@login_required
def get_user_deletion_info(user_id: int):
    """Return information about what will be deleted when removing a user."""
    if current_user.role != "admin":
        abort(403)

    user = _session_get_or_404(User, user_id)

    if user.role == "sender":
        entries = ConversationEntry.query.filter_by(sender_id=user.id).all()
        total_entries = len(entries)
        non_ready_entries = sum(
            1 for entry in entries if entry.status != ENTRY_STATUS_READY
        )

        # Check if this sender is also a final sender
        final_sender_entries = ConversationEntry.query.filter_by(
            final_sender_id=user.id,
        ).count()

        return jsonify(
            role="sender",
            total_entries=total_entries,
            non_ready_entries=non_ready_entries,
            final_sender_entries=final_sender_entries,
        )

    if user.role == "receiver":
        # Count entries assigned to this receiver
        entries = ConversationEntry.query.filter_by(receiver_id=user.id).all()
        entry_ids = [entry.id for entry in entries]

        # Count total ReceiverResponse instances across all their entries
        total_responses = 0
        if entry_ids:
            total_responses = ReceiverResponse.query.filter(
                ReceiverResponse.conversation_entry_id.in_(entry_ids)
            ).count()

        # Count entries that will be rolled back (those in 'evaluated' status)
        entries_to_rollback = sum(
            1 for entry in entries if entry.status == ENTRY_STATUS_EVALUATED
        )

        return jsonify(
            role="receiver",
            total_entries=len(entries),
            total_responses=total_responses,
            entries_to_rollback=entries_to_rollback,
        )

    if user.role == "validator":
        # Count entries assigned to this validator
        entries = ConversationEntry.query.filter_by(validator_id=user.id).all()
        entry_ids = [entry.id for entry in entries]

        # Count total ValidatorResponseEvaluation instances across all their entries
        total_evaluations = 0
        if entry_ids:
            total_evaluations = ValidatorResponseEvaluation.query.filter(
                ValidatorResponseEvaluation.conversation_entry_id.in_(entry_ids)
            ).count()

        # Count entries that will be rolled back (those in 'validated' status)
        entries_to_rollback = sum(
            1 for entry in entries if entry.status == ENTRY_STATUS_VALIDATED
        )

        return jsonify(
            role="validator",
            total_entries=len(entries),
            total_evaluations=total_evaluations,
            entries_to_rollback=entries_to_rollback,
        )

    # For admin or unknown roles
    return jsonify(
        role=user.role,
        total_entries=0,
    )


@app.route("/terms", methods=["GET", "POST"])
@login_required
def terms():
    if current_user.role == "admin":
        return redirect(get_post_login_redirect(current_user))

    terms_required = is_terms_enforcement_enabled()

    if current_user.terms_accepted:
        flash("You already accepted the terms and conditions.", "info")
        return redirect(get_post_login_redirect(current_user))

    if request.method == "POST":
        try:
            current_user.mark_terms_accepted()
            db.session.commit()
            logger.info(f"User {current_user.unique_id} accepted terms")
        except SQLAlchemyError as exc:
            db.session.rollback()
            logger.exception(
                f"Failed to record terms acceptance for user {current_user.unique_id}: {exc}"
            )
            flash("We could not record your acceptance. Please try again.", "danger")
            return render_template(
                "terms.html",
                user_role=current_user.role,
                terms_required=terms_required,
            )

        flash("Thank you for accepting the study terms.", "success")
        return redirect(get_post_login_redirect(current_user))

    return render_template(
        "terms.html",
        user_role=current_user.role,
        terms_required=terms_required,
    )


@app.route("/confirm-instructions", methods=["POST"])
@login_required
def confirm_instructions():
    """Mark the current user's instructions as read."""
    if current_user.role == "admin":
        return jsonify(success=False, error="Admins do not have instructions"), 400

    if current_user.instructions_read:
        return jsonify(success=True, already_confirmed=True)

    try:
        current_user.mark_instructions_read()
        db.session.commit()
        logger.info(f"User {current_user.unique_id} marked instructions as read")
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(
            f"Failed to record instructions confirmation for user {current_user.unique_id}: {exc}"
        )
        return jsonify(success=False, error="Database error"), 500

    return jsonify(success=True)


@app.route("/home")
@login_required
def participant_home():
    user_obj = cast(User, current_user)
    if user_obj.role == "admin":
        return redirect(url_for("dashboard"))
    receiver_locale = get_receiver_locale_for_user(user_obj)

    def receiver_text(key: str, default: str, **kwargs: Any) -> str:
        if receiver_locale:
            return receiver_locale.text(key, default, **kwargs)
        if kwargs:
            try:
                return default.format(**kwargs)
            except KeyError:
                return default
        return default

    entries = (
        ConversationEntry.query.filter(
            or_(
                ConversationEntry.sender_id == user_obj.id,
                ConversationEntry.final_sender_id == user_obj.id,
                ConversationEntry.receiver_id == user_obj.id,
                ConversationEntry.validator_id == user_obj.id,
            )
        )
        .order_by(ConversationEntry.created_at.desc())
        .all()
    )

    # Determine whether this sender is acting as a final sender on any entries
    is_final_sender = any(
        entry.final_sender_id == user_obj.id for entry in entries
    ) if user_obj.role == "sender" else False

    entry_rows = []
    for entry in entries:
        if entry.final_sender_id == user_obj.id:
            participant_role = "FinalSender"
        elif entry.sender_id == user_obj.id:
            participant_role = "Sender"
        elif entry.receiver_id == user_obj.id:
            participant_role = "Receiver"
        elif entry.validator_id == user_obj.id:
            participant_role = "Validator"
        else:
            participant_role = "Participant"

        conversation = entry.conversation
        status_value = entry.status or ENTRY_STATUS_READY

        action_url = None
        is_clickable = False
        action_label = "Start"

        if participant_role == "FinalSender":
            if status_value == ENTRY_STATUS_VALIDATED:
                action_url = url_for("sender_final_survey", entry_id=entry.id)
                is_clickable = True
                action_label = "Complete survey"
            elif status_value == ENTRY_STATUS_COMPLETED:
                action_label = "Completed"
            else:
                action_label = "Awaiting review"
        elif participant_role == "Sender":
            if status_value == ENTRY_STATUS_READY:
                action_url = url_for("sender_annotation", entry_id=entry.id)
                is_clickable = True
                action_label = "Start"
            elif status_value == ENTRY_STATUS_COMPLETED:
                action_label = "Completed"
            else:
                action_label = "Awaiting review"
        elif participant_role == "Receiver":
            if status_value == ENTRY_STATUS_TRANSLATED:
                action_url = url_for("receiver_annotation", entry_id=entry.id)
                is_clickable = True
                action_label = (
                    receiver_text("home.button.start", "Start")
                    if not entry.receiver_responses
                    else receiver_text("home.button.continue", "Continue")
                )
            else:
                if status_value in {ENTRY_STATUS_READY, ENTRY_STATUS_RECORDED}:
                    action_label = receiver_text(
                        "home.button.awaiting",
                        "Awaiting translation",
                    )
                else:
                    action_label = receiver_text(
                        "home.button.completed",
                        "Completed",
                    )
        elif participant_role == "Validator":
            if status_value in {
                ENTRY_STATUS_EVALUATED,
                ENTRY_STATUS_VALIDATED,
                ENTRY_STATUS_COMPLETED,
            }:
                action_url = url_for("validator_annotation", entry_id=entry.id)
                is_clickable = True
                if status_value == ENTRY_STATUS_EVALUATED:
                    action_label = "Continue" if entry.validator_review else "Start"
                else:
                    action_label = "View"
            else:
                action_label = "Pending"
        else:
            action_label = "View"

        entry_rows.append(
            {
                "conversation_id": (conversation.external_id if conversation else "—"),
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "entry_id": entry.id,
                "status": status_value,
                "participant_role": participant_role,
                "status_label": (
                    receiver_text(
                        f"status.{status_value}",
                        ENTRY_STATUS_LABELS.get(
                            status_value,
                            status_value.replace("_", " ").title(),
                        ),
                    )
                    if participant_role == "Receiver"
                    else ENTRY_STATUS_LABELS.get(
                        status_value,
                        status_value.replace("_", " ").title(),
                    )
                ),
                "action_url": action_url,
                "is_clickable": is_clickable,
                "action_label": action_label,
                "requires_survey": (
                    participant_role == "FinalSender"
                    and status_value == ENTRY_STATUS_VALIDATED
                ),
            }
        )

    progress = _calculate_user_progress(user_obj)

    # Determine which confirmation code to show
    show_confirmation_code = False
    display_confirmation_code = None

    if user_obj.role == "sender":
        if is_final_sender:
            # Final sender: show code when all final_sender entries are completed
            final_sender_entries = [
                row for row in entry_rows
                if row.get("participant_role") == "FinalSender"
            ]
            if final_sender_entries:
                all_completed = all(
                    row["status"] == ENTRY_STATUS_COMPLETED
                    for row in final_sender_entries
                )
                if all_completed and user_obj.confirmation_code:
                    show_confirmation_code = True
                    display_confirmation_code = user_obj.confirmation_code
        else:
            # Original sender (phase 1 only): show code when all entries recorded
            sender_entries = [row for row in entry_rows]
            if sender_entries:
                all_recorded = all(
                    row["status"] != ENTRY_STATUS_READY for row in sender_entries
                )
                if all_recorded and user_obj.confirmation_code:
                    show_confirmation_code = True
                    display_confirmation_code = user_obj.confirmation_code
    else:
        # Receivers and validators: original logic
        if progress and progress.get("total"):
            all_done = progress.get("completed", 0) >= progress.get("total", 0)
            if all_done and user_obj.confirmation_code:
                show_confirmation_code = True
                display_confirmation_code = user_obj.confirmation_code

    total_entries = len(entry_rows)
    current_role_label = user_obj.role.capitalize()
    if receiver_locale and user_obj.role == "receiver":
        current_role_label = receiver_locale.text(
            "roles.receiver",
            current_role_label,
        )

    return render_template(
        "participant_home.html",
        entries=entry_rows,
        total_entries=total_entries,
        progress=progress,
        receiver_locale=receiver_locale,
        current_role_label=current_role_label,
        instructions_read=user_obj.instructions_read,
        show_confirmation_code=show_confirmation_code,
        confirmation_code=display_confirmation_code,
        is_final_sender=is_final_sender,
    )


def _ensure_recordings_dir() -> Path:
    recordings_dir = Path(app.config["RECORDINGS_DIR"])
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir


def _get_next_sender_entry(
    current_entry: ConversationEntry, user_id: int
) -> Optional[ConversationEntry]:
    return (
        ConversationEntry.query.filter(
            ConversationEntry.sender_id == user_id,
            ConversationEntry.status.in_([ENTRY_STATUS_READY, ENTRY_STATUS_VALIDATED]),
            or_(
                ConversationEntry.created_at > current_entry.created_at,
                and_(
                    ConversationEntry.created_at == current_entry.created_at,
                    ConversationEntry.id > current_entry.id,
                ),
            ),
        )
        .order_by(ConversationEntry.created_at.asc(), ConversationEntry.id.asc())
        .first()
    )


def _get_next_receiver_entry(
    current_entry: ConversationEntry, user_id: int
) -> Optional[ConversationEntry]:
    return (
        ConversationEntry.query.filter(
            ConversationEntry.receiver_id == user_id,
            ConversationEntry.status.in_(
                [ENTRY_STATUS_TRANSLATED, ENTRY_STATUS_EVALUATED]
            ),
            or_(
                ConversationEntry.created_at > current_entry.created_at,
                and_(
                    ConversationEntry.created_at == current_entry.created_at,
                    ConversationEntry.id > current_entry.id,
                ),
            ),
        )
        .order_by(ConversationEntry.created_at.asc(), ConversationEntry.id.asc())
        .first()
    )


def _get_next_validator_entry(
    current_entry: ConversationEntry, user_id: int
) -> Optional[ConversationEntry]:
    return (
        ConversationEntry.query.filter(
            ConversationEntry.validator_id == user_id,
            ConversationEntry.status.in_(
                [ENTRY_STATUS_EVALUATED, ENTRY_STATUS_VALIDATED]
            ),
            or_(
                ConversationEntry.created_at > current_entry.created_at,
                and_(
                    ConversationEntry.created_at == current_entry.created_at,
                    ConversationEntry.id > current_entry.id,
                ),
            ),
        )
        .order_by(ConversationEntry.created_at.asc(), ConversationEntry.id.asc())
        .first()
    )


def _get_app_setting_value(key: str, default: Optional[str] = None) -> Optional[str]:
    _ensure_app_settings_table()
    # Import AppSetting locally to avoid import-time circular references
    from .models import AppSetting

    setting = AppSetting.query.filter_by(key=key).first()
    if setting is None:
        return default
    return setting.value


def _set_app_setting_value(key: str, value: str) -> None:
    _ensure_app_settings_table()
    setting = AppSetting.query.filter_by(key=key).first()
    if setting:
        setting.value = value
    else:
        db.session.add(AppSetting(key=key, value=value))
    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception("Failed to persist app setting %s", key, exc_info=exc)
        raise


def is_terms_enforcement_enabled() -> bool:
    value = _get_app_setting_value(TERMS_ENFORCEMENT_SETTING_KEY, "0") or "0"
    return value == "1"


def set_terms_enforcement_enabled(enabled: bool) -> None:
    _set_app_setting_value(TERMS_ENFORCEMENT_SETTING_KEY, "1" if enabled else "0")


def is_enforce_same_gender_enabled() -> bool:
    value = _get_app_setting_value(ENFORCE_SAME_GENDER_KEY, "0") or "0"
    return value == "1"


def set_enforce_same_gender_enabled(enabled: bool) -> None:
    _set_app_setting_value(ENFORCE_SAME_GENDER_KEY, "1" if enabled else "0")


def is_maintenance_enabled() -> bool:
    value = _get_app_setting_value(MAINTENANCE_MODE_KEY, "0") or "0"
    return value == "1"


def set_maintenance_enabled(enabled: bool) -> None:
    _set_app_setting_value(MAINTENANCE_MODE_KEY, "1" if enabled else "0")


def _ensure_app_settings_table() -> None:
    global _SETTINGS_TABLE_READY
    if _SETTINGS_TABLE_READY:
        return

    # AppSetting is defined in models.py; import here to avoid circular imports at module import time
    from .models import AppSetting

    try:
        AppSetting.__table__.create(bind=db.engine, checkfirst=True)  # type: ignore[attr-defined]
    except SQLAlchemyError as exc:
        app.logger.exception(
            "Failed to ensure app_settings table exists",
            exc_info=exc,
        )
    else:
        _SETTINGS_TABLE_READY = True


def _load_receiver_locale_from_file(code: str) -> Optional[ReceiverLocalization]:
    locale_path = LOCALES_DIR / f"{code}.json"
    if not locale_path.exists():
        return None
    try:
        payload = json.loads(locale_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        app.logger.warning(
            "Failed to decode receiver locale file %s: %s",
            locale_path,
            exc,
        )
        return None

    strings = payload.get("strings")
    if not isinstance(strings, dict):
        app.logger.warning(
            "Locale file %s missing strings map; skipping",
            locale_path,
        )
        return None

    code_value = payload.get("code") or code
    label_value = payload.get("label") or code_value.upper()
    return ReceiverLocalization(code_value, label_value, strings)


def list_receiver_locale_options() -> list[dict[str, str]]:
    if not LOCALES_DIR.exists():
        return []

    items: list[dict[str, str]] = []
    for path in LOCALES_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        code_value = str(payload.get("code") or path.stem)
        label_value = str(payload.get("label") or code_value.upper())
        items.append({"code": code_value, "label": label_value})

    items.sort(key=lambda item: item["label"].lower())
    return items


def _get_receiver_locale_code() -> str:
    stored = (_get_app_setting_value(RECEIVER_UI_LOCALE_KEY) or "").strip().lower()
    if stored:
        return stored
    return DEFAULT_RECEIVER_UI_LOCALE


def set_receiver_locale(code: str) -> None:
    normalized = code.strip().lower() or DEFAULT_RECEIVER_UI_LOCALE
    _set_app_setting_value(RECEIVER_UI_LOCALE_KEY, normalized)
    _RECEIVER_LOCALE_CACHE.clear()


def get_receiver_locale(code: Optional[str] = None) -> Optional[ReceiverLocalization]:
    target = (code or _get_receiver_locale_code()).strip().lower()
    if not target:
        target = DEFAULT_RECEIVER_UI_LOCALE

    if target not in _RECEIVER_LOCALE_CACHE:
        locale = _load_receiver_locale_from_file(target)
        if locale is None and target != DEFAULT_RECEIVER_UI_LOCALE:
            locale = _load_receiver_locale_from_file(DEFAULT_RECEIVER_UI_LOCALE)
        if locale:
            _RECEIVER_LOCALE_CACHE[target] = locale
        elif target != DEFAULT_RECEIVER_UI_LOCALE:
            fallback = _load_receiver_locale_from_file(DEFAULT_RECEIVER_UI_LOCALE)
            if fallback:
                _RECEIVER_LOCALE_CACHE[target] = fallback

    locale_obj = _RECEIVER_LOCALE_CACHE.get(target)
    if locale_obj is None and target != DEFAULT_RECEIVER_UI_LOCALE:
        locale_obj = _RECEIVER_LOCALE_CACHE.get(DEFAULT_RECEIVER_UI_LOCALE)
    return locale_obj


def get_receiver_locale_for_user(user: User) -> Optional[ReceiverLocalization]:
    if user.role != "receiver":
        return None
    return get_receiver_locale()


def _calculate_user_progress(user: "User") -> Optional[dict[str, int]]:
    if user.role == "sender":
        # Check if this sender is also a final sender
        final_sender_entries = ConversationEntry.query.filter_by(
            final_sender_id=user.id
        ).all()
        if final_sender_entries:
            # Final sender: progress based on completing surveys
            total = len(final_sender_entries)
            completed = sum(
                1
                for entry in final_sender_entries
                if entry.status == ENTRY_STATUS_COMPLETED
            )
        else:
            # Original sender: progress based on recording
            assignments = ConversationEntry.query.filter_by(sender_id=user.id).all()
            total = len(assignments)
            completed = sum(
                1
                for entry in assignments
                if entry.status
                and entry.status != ENTRY_STATUS_READY
            )
    elif user.role == "receiver":
        assignments = ConversationEntry.query.filter(
            ConversationEntry.receiver_id == user.id,
            ConversationEntry.status.in_(
                [ENTRY_STATUS_TRANSLATED, ENTRY_STATUS_EVALUATED]
            ),
        ).all()
        total = len(assignments)
        completed = sum(
            1 for entry in assignments if entry.status == ENTRY_STATUS_EVALUATED
        )
    elif user.role == "validator":
        assignments = ConversationEntry.query.filter(
            ConversationEntry.validator_id == user.id,
            ConversationEntry.status.in_(
                [
                    ENTRY_STATUS_EVALUATED,
                    ENTRY_STATUS_VALIDATED,
                    ENTRY_STATUS_COMPLETED,
                ]
            ),
        ).all()
        total = len(assignments)
        completed = sum(
            1
            for entry in assignments
            if entry.status in {ENTRY_STATUS_VALIDATED, ENTRY_STATUS_COMPLETED}
            or entry.validator_review is not None
        )
    else:
        return None

    percent = int(round((completed / total) * 100)) if total else 0

    return {
        "total": total,
        "completed": completed,
        "percent": percent,
    }


def _generate_confirmation_code() -> str:
    """Generate a unique 4-byte hexadecimal confirmation code."""
    for _ in range(10):  # Attempt up to 10 times to find a unique code
        code = secrets.token_hex(4)  # 4 bytes = 8 hex characters
        if not User.query.filter_by(confirmation_code=code).first():
            return code
    # Fallback: append timestamp if collision persists
    return secrets.token_hex(4)


def _check_and_generate_sender_recording_code(user: "User") -> Optional[str]:
    """Check if sender has completed all recordings and generate confirmation code.

    All recordings are done when every entry assigned to this sender has moved
    past 'ready' status (i.e., recorded or later).
    Returns the confirmation code if newly generated, or None if already exists or not complete.
    """
    if user.role != "sender":
        return None

    if user.confirmation_code:
        logger.debug(f"Sender {user.unique_id} already has confirmation code")
        return None

    assignments = ConversationEntry.query.filter_by(sender_id=user.id).all()
    if not assignments:
        return None

    # All recordings done when all entries are at least 'recorded' (not 'ready')
    all_recorded = all(entry.status != ENTRY_STATUS_READY for entry in assignments)
    if not all_recorded:
        return None

    # Assign the predefined sender confirmation code
    sender_code = ROLE_CONFIRMATION_CODES.get("sender")
    if not sender_code:
        logger.warning(
            "No confirmation code configured for sender role; skipping code assignment"
        )
        return None

    user.confirmation_code = sender_code
    logger.info(
        f"Assigned recording confirmation code for sender {user.unique_id}: {user.confirmation_code}"
    )
    return user.confirmation_code


def _check_and_generate_completion_code(user: "User") -> Optional[str]:
    """Check if user has completed all tasks and generate confirmation code if so.

    For senders acting as final senders, completion means all their final_sender
    entries are COMPLETED. For receivers/validators the original logic applies.
    Returns the confirmation code if newly generated, or None if already exists or not complete.
    """
    if user.confirmation_code:
        logger.debug(f"User {user.unique_id} already has confirmation code")
        return None  # Already has a code

    if user.role == "sender":
        # Final senders get code when all entries where they are final_sender are COMPLETED
        assignments = ConversationEntry.query.filter_by(
            final_sender_id=user.id,
        ).all()
        if not assignments:
            return None
        all_completed = all(
            entry.status == ENTRY_STATUS_COMPLETED for entry in assignments
        )
        if not all_completed:
            return None
        # Use the "final_sender" confirmation code
        role_code = ROLE_CONFIRMATION_CODES.get("final_sender")
    elif user.role == "receiver":
        # Receivers get code when all their assigned entries are EVALUATED
        assignments = ConversationEntry.query.filter(
            ConversationEntry.receiver_id == user.id,
            ConversationEntry.status.in_(
                [ENTRY_STATUS_TRANSLATED, ENTRY_STATUS_EVALUATED]
            ),
        ).all()
        if not assignments:
            return None
        all_completed = all(
            entry.status == ENTRY_STATUS_EVALUATED for entry in assignments
        )
        if not all_completed:
            return None
        role_code = ROLE_CONFIRMATION_CODES.get("receiver")
    elif user.role == "validator":
        # Validators get code when all their assigned entries have been validated
        assignments = ConversationEntry.query.filter(
            ConversationEntry.validator_id == user.id,
            ConversationEntry.status.in_(
                [ENTRY_STATUS_EVALUATED, ENTRY_STATUS_VALIDATED, ENTRY_STATUS_COMPLETED]
            ),
        ).all()
        if not assignments:
            return None
        all_completed = all(
            entry.status in {ENTRY_STATUS_VALIDATED, ENTRY_STATUS_COMPLETED}
            or entry.validator_review is not None
            for entry in assignments
        )
        if not all_completed:
            return None
        role_code = ROLE_CONFIRMATION_CODES.get("validator")
    else:
        return None

    # Assign the predefined role-specific confirmation code
    if not role_code:
        logger.warning(
            f"No confirmation code configured for role '{user.role}'; skipping code assignment"
        )
        return None
    user.confirmation_code = role_code
    logger.info(
        f"Assigned completion confirmation code for user {user.unique_id} (role={user.role}): {user.confirmation_code}"
    )
    return user.confirmation_code


@app.route("/entries/<int:entry_id>/annotate")
@login_required
def sender_annotation(entry_id: int):
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.sender_id != current_user.id:
        abort(403)

    if entry.status == ENTRY_STATUS_VALIDATED:
        return redirect(url_for("sender_final_survey", entry_id=entry.id))

    if entry.status != ENTRY_STATUS_READY:
        abort(403)

    next_entry = _get_next_sender_entry(entry, current_user.id)
    recording = entry.recording
    progress = _calculate_user_progress(cast(User, current_user))

    return render_template(
        "sender_annotation.html",
        entry=entry,
        conversation=entry.conversation,
        recording=recording,
        next_entry=next_entry,
        progress=progress,
    )


@app.route("/entries/<int:entry_id>/final-survey", methods=["GET", "POST"])
@login_required
def sender_final_survey(entry_id: int):
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.final_sender_id != current_user.id:
        abort(403)

    if entry.status == ENTRY_STATUS_COMPLETED:
        flash(
            "You already completed the final survey for this conversation.",
            "info",
        )
        return redirect(url_for("participant_home"))

    if entry.status != ENTRY_STATUS_VALIDATED:
        flash(
            "This conversation is not ready for the final survey yet.",
            "warning",
        )
        return redirect(url_for("participant_home"))

    survey = entry.final_survey or SenderFinalSurvey(
        conversation_entry_id=entry.id,
        sender_id=current_user.id,
    )

    # We had a "Usefulness" score among the scores to be verified. We are leaving this 
    # and all the relevant part below commented out as an example of where 
    # and how a new field could be added.
    survey_step = "baseline"
    if (
        # survey.baseline_usefulness_score is not None
        survey.baseline_satisfaction_score is not None
        and survey.baseline_trust_score is not None
        and survey.baseline_reliance_score is not None
    ):
        survey_step = "final"

    if request.method == "POST":
        form_step = request.form.get("survey_step")
        if form_step == "baseline":
            # raw_usefulness_score = request.form.get("baseline_usefulness_score")
            raw_satisfaction_score = request.form.get("baseline_satisfaction_score")
            raw_trust_score = request.form.get("baseline_trust_score")
            raw_reliance_score = request.form.get("baseline_reliance_score")
            try:
                # baseline_usefulness_score = (
                #     int(raw_usefulness_score)
                #     if raw_usefulness_score is not None
                #     else None
                # )
                baseline_satisfaction_score = (
                    int(raw_satisfaction_score)
                    if raw_satisfaction_score is not None
                    else None
                )
                baseline_trust_score = (
                    int(raw_trust_score) if raw_trust_score is not None else None
                )
                baseline_reliance_score = (
                    int(raw_reliance_score) if raw_reliance_score is not None else None
                )
            except (TypeError, ValueError):
                # baseline_usefulness_score = None
                baseline_satisfaction_score = None
                baseline_trust_score = None
                baseline_reliance_score = None

            if (
                # baseline_usefulness_score is None
                baseline_satisfaction_score is None
                or baseline_trust_score is None
                or baseline_reliance_score is None
            ):
                flash(
                    "Please choose a response for all questions before submitting the survey.",
                    "danger",
                )
            elif (
                # baseline_usefulness_score < 1
                # or baseline_usefulness_score > 5
                baseline_satisfaction_score < 1
                or baseline_satisfaction_score > 5
                or baseline_trust_score < 1
                or baseline_trust_score > 5
                or baseline_reliance_score < 1
                or baseline_reliance_score > 5
            ):
                flash(
                    "Please choose a number between 1 and 5 for all questions.",
                    "danger",
                )
            else:
                # survey.baseline_usefulness_score = baseline_usefulness_score
                survey.baseline_satisfaction_score = baseline_satisfaction_score
                survey.baseline_trust_score = baseline_trust_score
                survey.baseline_reliance_score = baseline_reliance_score
                if not survey.id:
                    db.session.add(survey)
                try:
                    db.session.commit()
                    logger.info(
                        "Sender %s completed baseline survey for entry %s (satisfaction: %s, trust: %s, reliance: %s)",
                        current_user.unique_id,
                        entry_id,
                        # baseline_usefulness_score,
                        baseline_satisfaction_score,
                        baseline_trust_score,
                        baseline_reliance_score,
                    )
                    flash(
                        "Thank you for your initial feedback. Now, please review the additional information below and answer the final questions.",
                        "success",
                    )
                    return redirect(url_for("sender_final_survey", entry_id=entry_id))
                except SQLAlchemyError as exc:
                    db.session.rollback()
                    logger.exception(
                        f"Failed to save baseline survey for entry {entry_id}: {exc}"
                    )
                    flash(
                        "We could not save your response due to a database error. Please try again.",
                        "danger",
                    )

        elif form_step == "final":
            # raw_usefulness_score = request.form.get("usefulness_score")
            raw_satisfaction_score = request.form.get("satisfaction_score")
            raw_trust_score = request.form.get("trust_score")
            raw_reliance_score = request.form.get("reliance_score")
            try:
                # usefulness_score = (
                #     int(raw_usefulness_score)
                #     if raw_usefulness_score is not None
                #     else None
                # )
                satisfaction_score = (
                    int(raw_satisfaction_score)
                    if raw_satisfaction_score is not None
                    else None
                )
                trust_score = (
                    int(raw_trust_score) if raw_trust_score is not None else None
                )
                reliance_score = (
                    int(raw_reliance_score) if raw_reliance_score is not None else None
                )
            except (TypeError, ValueError):
                # usefulness_score = None
                satisfaction_score = None
                trust_score = None
                reliance_score = None

            if (
                # usefulness_score is None
                satisfaction_score is None
                or trust_score is None
                or reliance_score is None
            ):
                flash(
                    "Please choose a response for all questions before submitting the survey.",
                    "danger",
                )
            elif (
                # usefulness_score < 1
                # or usefulness_score > 5
                satisfaction_score < 1
                or satisfaction_score > 5
                or trust_score < 1
                or trust_score > 5
                or reliance_score < 1
                or reliance_score > 5
            ):
                flash(
                    "Please choose a number between 1 and 5 for all questions.",
                    "danger",
                )
            else:
                # survey.usefulness_score = usefulness_score
                survey.satisfaction_score = satisfaction_score
                survey.trust_score = trust_score
                survey.reliance_score = reliance_score
                entry.status = ENTRY_STATUS_COMPLETED
                # Check if all sender tasks are now complete and generate code
                user_obj = cast(User, current_user)
                _check_and_generate_completion_code(user_obj)
                try:
                    db.session.commit()
                    logger.info(
                        "Sender %s completed final survey for entry %s (satisfaction: %s, trust: %s, reliance: %s, status: %s)",
                        current_user.unique_id,
                        entry_id,
                        # usefulness_score,
                        satisfaction_score,
                        trust_score,
                        reliance_score,
                        entry.status,
                    )
                    flash("Thank you for completing the final survey. Please check below if there is any assignment left.", "success")
                    return redirect(url_for("participant_home"))
                except SQLAlchemyError as exc:
                    db.session.rollback()
                    logger.exception(
                        f"Failed to save final survey for entry {entry_id}: {exc}"
                    )
                    flash(
                        "We could not save your response due to a database error. Please try again.",
                        "danger",
                    )

    recording = entry.recording
    recording_url = (
        url_for("preview_entry_recording", entry_id=entry.id) if recording else None
    )

    translated_text = (entry.translation_text or "").strip()

    validator_checks = {
        evaluation.question_id: evaluation.is_correct
        for evaluation in entry.validator_evaluations
    }

    questions = (
        Question.query.filter_by(conversation_id=entry.conversation_id)
        .order_by(Question.id.asc())
        .all()
    )

    question_items: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        text_q = (question.question_text or "").strip()
        question_items.append(
            {
                "id": question.id,
                "number": index,
                "text": text_q,
                "validator_correct": validator_checks.get(question.id),
            }
        )

    validator_review = entry.validator_review
    quality_score = validator_review.translation_score if validator_review else None
    progress = _calculate_user_progress(cast(User, current_user))

    translation_text = (entry.translation_text or "").strip()

    original_text = entry.conversation.source_text or ""

    return render_template(
        "final_survey.html",
        entry=entry,
        phase=survey_step,
        survey=survey,
        conversation_text=original_text,
        translated_text=translated_text,
        questions=question_items,
    )


@app.route("/entries/<int:entry_id>/annotate/receiver")
@login_required
def receiver_annotation(entry_id: int):
    entry = _session_get_or_404(ConversationEntry, entry_id)
    user_obj = cast(User, current_user)

    if entry.receiver_id != user_obj.id:
        abort(403)

    if entry.status not in {ENTRY_STATUS_TRANSLATED, ENTRY_STATUS_EVALUATED}:
        abort(403)

    receiver_locale = get_receiver_locale_for_user(user_obj)

    def receiver_text(key: str, default: str, **kwargs: Any) -> str:
        if receiver_locale:
            return receiver_locale.text(key, default, **kwargs)
        if kwargs:
            try:
                return default.format(**kwargs)
            except KeyError:
                return default
        return default

    translation_text = (entry.translation_text or "").strip()

    questions = (
        Question.query.filter_by(conversation_id=entry.conversation_id)
        .order_by(Question.id.asc())
        .all()
    )

    responses_by_question_id = {
        response.question_id: response.response_text or ""
        for response in entry.receiver_responses
    }

    question_items = []
    for index, question in enumerate(questions, start=1):
        translated_text = (
            question.translated_question or question.question_text or ""
        ).strip()
        question_items.append(
            {
                "id": question.id,
                "number": index,
                "translated_question": translated_text,
                "response": responses_by_question_id.get(question.id, ""),
            }
        )

    has_responses = bool(entry.receiver_responses)
    is_complete = entry.status == ENTRY_STATUS_EVALUATED
    is_locked = entry.status in {
        ENTRY_STATUS_EVALUATED,
        ENTRY_STATUS_VALIDATED,
        ENTRY_STATUS_COMPLETED,
    }
    next_entry = _get_next_receiver_entry(entry, user_obj.id)
    progress = _calculate_user_progress(user_obj)

    if is_locked:
        initial_status_message = receiver_text(
            "annotation.status.locked",
            "This submission has been finalized. Responses are read-only.",
        )
    elif has_responses and is_complete:
        initial_status_message = receiver_text(
            "annotation.status.loaded",
            "Your previously saved responses are loaded. You can review them below.",
        )
    elif has_responses:
        initial_status_message = receiver_text(
            "annotation.status.draft_loaded",
            "Draft responses are loaded. Please complete every answer before submitting.",
        )
    else:
        initial_status_message = receiver_text(
            "annotation.status.prompt",
            "Answer each question and click Save to record your responses.",
        )

    save_button_label = receiver_text(
        "annotation.buttons.save",
        "Save responses",
    )
    next_button_label = receiver_text(
        "annotation.buttons.next" if next_entry else "annotation.buttons.back",
        "Go to next assignment" if next_entry else "Back to assignments",
    )

    receiver_locale_strings: dict[str, str] = (
        receiver_locale.to_dict() if receiver_locale else {}
    )

    return render_template(
        "receiver_annotation.html",
        entry=entry,
        conversation=entry.conversation,
        question_items=question_items,
        has_responses=has_responses,
        is_complete=is_complete,
        is_locked=is_locked,
        next_entry=next_entry,
        progress=progress,
        translation_text=translation_text,
        receiver_locale=receiver_locale,
        receiver_locale_strings=receiver_locale_strings,
        initial_status_message=initial_status_message,
        save_button_label=save_button_label,
        next_button_label=next_button_label,
    )


@app.route("/entries/<int:entry_id>/recording/preview")
@login_required
def preview_entry_recording(entry_id: int):
    entry = _session_get_or_404(ConversationEntry, entry_id)

    allowed = False
    if current_user.role == "admin":
        allowed = True
    elif current_user.role == "validator" and entry.validator_id == current_user.id:
        allowed = True
    elif current_user.role == "sender" and (
        entry.sender_id == current_user.id
        or entry.final_sender_id == current_user.id
    ):
        allowed = True

    if not allowed:
        abort(403)

    recording = entry.recording
    if not recording:
        abort(404)

    recordings_dir = _ensure_recordings_dir().resolve()
    file_path = (recordings_dir / recording.stored_filename).resolve()

    try:
        file_path.relative_to(recordings_dir)
    except ValueError:
        abort(404)

    if not file_path.exists():
        abort(404)

    return send_file(
        file_path,
        mimetype=recording.file_mimetype or "audio/wav",
        download_name=recording.original_filename or file_path.name,
        as_attachment=False,
    )


@app.route("/entries/<int:entry_id>/annotate/validator")
@login_required
def validator_annotation(entry_id: int):
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.validator_id != current_user.id:
        abort(403)

    if entry.status not in {
        ENTRY_STATUS_EVALUATED,
        ENTRY_STATUS_VALIDATED,
        ENTRY_STATUS_COMPLETED,
    }:
        abort(403)

    # We have to use whatever translation text is present, even if empty
    translation_text = entry.translation_text.strip()

    receiver_responses = {
        response.question_id: (response.response_text or "").strip()
        for response in entry.receiver_responses
    }
    validator_checks = {
        evaluation.question_id: evaluation.is_correct
        for evaluation in entry.validator_evaluations
    }

    questions = (
        Question.query.filter_by(conversation_id=entry.conversation_id)
        .order_by(Question.id.asc())
        .all()
    )

    question_rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        question_rows.append(
            {
                "id": question.id,
                "number": index,
                "question_text": (question.question_text or "").strip(),
                "translated_question": (
                    question.translated_question or question.question_text or ""
                ).strip(),
                "reference_answer": (question.answer or "").strip(),
                "receiver_response": receiver_responses.get(question.id, ""),
                "is_correct": validator_checks.get(question.id),
            }
        )

    review = entry.validator_review
    translation_score = (
        round((review.translation_score / 100) * 6, 1) if review else None
    )
    audio_text_mismatch = bool(review.audio_text_mismatch) if review else False
    is_locked = entry.status in {
        ENTRY_STATUS_VALIDATED,
        ENTRY_STATUS_COMPLETED,
    }

    next_entry = _get_next_validator_entry(entry, current_user.id)
    progress = _calculate_user_progress(cast(User, current_user))

    recording_url = (
        url_for("preview_entry_recording", entry_id=entry.id)
        if entry.recording
        else None
    )

    return render_template(
        "validator_annotation.html",
        entry=entry,
        conversation=entry.conversation,
        translation_text=translation_text,
        translation_metadata={},
        question_rows=question_rows,
        translation_score=translation_score,
        audio_text_mismatch=audio_text_mismatch,
        validator_review=review,
        is_locked=is_locked,
        recording_url=recording_url,
        next_entry=next_entry,
        progress=progress,
    )


@app.route("/entries/<int:entry_id>/validator_review", methods=["POST"])
@login_required
def save_validator_review(entry_id: int):
    logger.info(
        f"Validator {current_user.unique_id} saving review for entry {entry_id}"
    )
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.validator_id != current_user.id:
        logger.warning(
            f"User {current_user.unique_id} attempted to access entry {entry_id} without permission"
        )
        abort(403)

    if entry.status in {
        ENTRY_STATUS_VALIDATED,
        ENTRY_STATUS_COMPLETED,
    }:
        logger.warning(
            f"Validator {current_user.unique_id} attempted to modify finalized entry {entry_id}"
        )
        return (
            jsonify(
                success=False,
                error="This validation is already finalized.",
            ),
            400,
        )

    payload = request.get_json(silent=True) or {}

    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    raw_score = payload.get("translation_score")
    if raw_score is None:
        return jsonify(success=False, error="Translation score is required."), 400

    try:
        translation_score = int(raw_score)
    except (TypeError, ValueError):
        return jsonify(success=False, error="Invalid translation score supplied."), 400

    answers_payload = payload.get("answers")
    if not isinstance(answers_payload, dict):
        return jsonify(success=False, error="Invalid answers payload."), 400

    questions = {
        question.id: question
        for question in (
            Question.query.filter_by(conversation_id=entry.conversation_id)
            .order_by(Question.id.asc())
            .all()
        )
    }

    if not questions:
        return (
            jsonify(
                success=False, error="No questions available for this conversation."
            ),
            400,
        )

    normalized_answers: dict[int, bool] = {}
    for raw_question_id, raw_value in answers_payload.items():
        try:
            question_id = int(raw_question_id)
        except (TypeError, ValueError):
            return (
                jsonify(success=False, error="Invalid question identifier supplied."),
                400,
            )

        if question_id not in questions:
            return (
                jsonify(success=False, error="Question does not belong to this entry."),
                400,
            )

        normalized_answers[question_id] = bool(raw_value)

    if set(normalized_answers.keys()) != set(questions.keys()):
        return (
            jsonify(
                success=False,
                error="A correctness decision is required for every question.",
            ),
            400,
        )

    review = entry.validator_review
    audio_text_mismatch = _coerce_bool(payload.get("audio_text_mismatch"))
    if review is None:
        review = ValidatorReview(
            conversation_entry_id=entry.id,
            validator_id=current_user.id,
            translation_score=translation_score,
            audio_text_mismatch=audio_text_mismatch,
        )
        db.session.add(review)
    else:
        review.translation_score = translation_score
        review.validator_id = current_user.id
        review.audio_text_mismatch = audio_text_mismatch

    existing_evaluations = {
        evaluation.question_id: evaluation for evaluation in entry.validator_evaluations
    }

    for question_id, is_correct in normalized_answers.items():
        evaluation = existing_evaluations.get(question_id)
        if evaluation is None:
            evaluation = ValidatorResponseEvaluation(
                conversation_entry_id=entry.id,
                question_id=question_id,
                is_correct=is_correct,
            )
            entry.validator_evaluations.append(evaluation)
            existing_evaluations[question_id] = evaluation
        else:
            evaluation.is_correct = is_correct

    if entry.status in {ENTRY_STATUS_EVALUATED, ENTRY_STATUS_VALIDATED}:
        entry.status = ENTRY_STATUS_VALIDATED

    # Check if all validator tasks are now complete and generate code
    user_obj = cast(User, current_user)
    _check_and_generate_completion_code(user_obj)

    try:
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        app.logger.exception("Failed to save validator review", exc_info=exc)
        return jsonify(success=False, error="Could not save validator feedback."), 500

    next_entry = _get_next_validator_entry(entry, current_user.id)

    return jsonify(
        success=True,
        next_url=(
            url_for("validator_annotation", entry_id=next_entry.id)
            if next_entry
            else url_for("participant_home")
        ),
    )


@app.route("/entries/<int:entry_id>/recordings", methods=["POST"])
@login_required
def save_sender_recording(entry_id: int):
    logger.info(
        f"Sender {current_user.unique_id} uploading recording for entry {entry_id}"
    )
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.sender_id != current_user.id:
        logger.warning(
            f"User {current_user.unique_id} attempted to upload recording to entry {entry_id} without permission"
        )
        abort(403)

    if entry.status not in {ENTRY_STATUS_READY, ENTRY_STATUS_VALIDATED}:
        logger.warning(
            f"Sender {current_user.unique_id} attempted to upload recording to already-submitted entry {entry_id}"
        )
        return (
            jsonify(success=False, error="This recording has already been submitted."),
            400,
        )

    file = request.files.get("audio_data")
    if file is None or file.filename == "":
        logger.warning(
            f"Sender {current_user.unique_id} submitted empty recording file for entry {entry_id}"
        )
        return jsonify(success=False, error="No audio file provided."), 400

    recordings_dir = _ensure_recordings_dir()
    entry_dir = recordings_dir / f"entry_{entry_id}"
    entry_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = file.filename or "recording.wav"
    original_filename = secure_filename(str(raw_filename)) or "recording.wav"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    stored_filename = f"entry_{entry_id}_{timestamp}.wav"
    save_path = entry_dir / stored_filename

    try:
        file.save(save_path)
        logger.info(
            f"Recording file saved for entry {entry_id}: {stored_filename} ({save_path.stat().st_size} bytes)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to save audio file for entry {entry_id}: {exc}")
        return jsonify(success=False, error="Could not save the audio file."), 500

    is_first_submission = entry.recording is None

    existing_recording = entry.recording
    if existing_recording:
        old_path = recordings_dir / existing_recording.stored_filename
        if old_path.exists():
            try:
                old_path.unlink()
            except OSError:
                app.logger.warning("Unable to delete old recording file: %s", old_path)

        existing_recording.stored_filename = str(save_path.relative_to(recordings_dir))
        existing_recording.original_filename = original_filename
        existing_recording.file_mimetype = file.mimetype
        existing_recording.file_size = save_path.stat().st_size
        existing_recording.created_at = datetime.now(timezone.utc)
    else:
        recording = Recording(
            conversation_entry_id=entry.id,
            stored_filename=str(save_path.relative_to(recordings_dir)),
            original_filename=original_filename,
            file_mimetype=file.mimetype,
            file_size=save_path.stat().st_size,
        )
        db.session.add(recording)

    entry.status = ENTRY_STATUS_RECORDED

    # Check if all sender recordings are now complete and generate confirmation code
    user_obj = cast(User, current_user)
    _check_and_generate_sender_recording_code(user_obj)

    try:
        db.session.commit()
        action = "updated" if not is_first_submission else "created"
        logger.info(
            f"Sender {current_user.unique_id} {action} recording for entry {entry_id} (status: {entry.status})"
        )
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(f"Failed to record audio metadata for entry {entry_id}: {exc}")
        return jsonify(success=False, error="Could not save recording data."), 500

    return jsonify(success=True, first_save=is_first_submission)


@app.route("/entries/<int:entry_id>/receiver_responses", methods=["POST"])
@login_required
def save_receiver_responses(entry_id: int):
    logger.info(
        f"Receiver {current_user.unique_id} saving responses for entry {entry_id}"
    )
    entry = _session_get_or_404(ConversationEntry, entry_id)

    if entry.receiver_id != current_user.id:
        logger.warning(
            f"User {current_user.unique_id} attempted to save responses to entry {entry_id} without permission"
        )
        abort(403)

    if entry.status in {
        ENTRY_STATUS_EVALUATED,
        ENTRY_STATUS_VALIDATED,
        ENTRY_STATUS_COMPLETED,
    }:
        logger.warning(
            f"Receiver {current_user.unique_id} attempted to modify finalized entry {entry_id}"
        )
        return (
            jsonify(
                success=False,
                error="Responses for this conversation are already finalized.",
            ),
            400,
        )

    payload = request.get_json(silent=True) or {}
    responses_payload = payload.get("responses")

    if not isinstance(responses_payload, dict):
        return jsonify(success=False, error="Invalid responses payload."), 400

    questions = {
        question.id: question
        for question in (
            Question.query.filter_by(conversation_id=entry.conversation_id)
            .order_by(Question.id.asc())
            .all()
        )
    }

    if not questions:
        return (
            jsonify(
                success=False, error="No questions available for this conversation."
            ),
            400,
        )

    normalized_payload: dict[int, str] = {}
    for key, value in responses_payload.items():
        try:
            question_id = int(key)
        except (TypeError, ValueError):
            return (
                jsonify(success=False, error="Invalid question identifier supplied."),
                400,
            )

        if question_id not in questions:
            return (
                jsonify(
                    success=False,
                    error="Question does not belong to this conversation.",
                ),
                400,
            )

        normalized_payload[question_id] = value if isinstance(value, str) else ""

    existing_responses = {
        response.question_id: response for response in entry.receiver_responses
    }

    missing_questions = [
        question_id
        for question_id in questions
        if not (normalized_payload.get(question_id, "").strip())
    ]

    if missing_questions:
        return (
            jsonify(
                success=False,
                error="Every question requires an answer before submitting.",
            ),
            400,
        )

    for question_id in questions:
        raw_value = normalized_payload.get(question_id, "")
        cleaned_value = raw_value.strip()

        response_obj = existing_responses.get(question_id)
        if response_obj is None:
            response_obj = ReceiverResponse(
                conversation_entry_id=entry.id,
                question_id=question_id,
                response_text=cleaned_value,
            )
            entry.receiver_responses.append(response_obj)
            existing_responses[question_id] = response_obj
        else:
            response_obj.response_text = cleaned_value

    all_responses_filled = all(
        (normalized_payload.get(question_id, "").strip()) for question_id in questions
    )

    if all_responses_filled:
        entry.status = ENTRY_STATUS_EVALUATED

    elif entry.status == ENTRY_STATUS_EVALUATED:
        entry.status = ENTRY_STATUS_TRANSLATED

    # Check if all receiver tasks are now complete and generate code
    user_obj = cast(User, current_user)
    _check_and_generate_completion_code(user_obj)

    try:
        db.session.commit()
        logger.info(
            f"Receiver {current_user.unique_id} saved responses for entry {entry_id} (status: {entry.status})"
        )
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception(
            f"Failed to save receiver responses for entry {entry_id}: {exc}"
        )
        return jsonify(success=False, error="Could not save responses."), 500

    next_entry = _get_next_receiver_entry(entry, current_user.id)

    has_responses = bool(entry.receiver_responses)

    return jsonify(
        success=True,
        next_url=(
            url_for("receiver_annotation", entry_id=next_entry.id)
            if next_entry
            else url_for("participant_home")
        ),
        has_responses=has_responses,
        complete=all_responses_filled,
    )


# ---------------------------------------------------------------------------
# Admin UI Preview Routes
# ---------------------------------------------------------------------------


class MockConversation:
    """Mock conversation object for admin UI previews."""

    external_id = "DEMO-001"
    topic = "Booking a hotel room"
    source = "Sample Dataset"
    source_text = """Good morning! I'd like to book a room for two nights, please.
Of course! We have several options available. Would you prefer a standard room or a suite?
A standard room would be fine. Does it have a view?
Yes, we have rooms with city views or garden views. The garden view is slightly quieter.
I'll take the garden view, please. What's the rate per night?
That would be €120 per night, including breakfast. Would you like me to proceed with the booking?
Yes, please. Can I also request a late checkout on the second day?
Certainly! Late checkout until 2 PM is available for an additional €30. Shall I add that?
Yes, that would be perfect. Thank you for your help!"""


class MockEntry:
    """Mock conversation entry for admin UI previews."""

    id = 0
    status = "translated"
    created_at = datetime.now(timezone.utc)
    conversation_id = 1
    conversation = MockConversation()


class MockRecording:
    """Mock recording presence indicator."""

    pass


class MockQuestion:
    """Mock question for admin UI previews."""

    def __init__(
        self,
        qid: int,
        number: int,
        text: str,
        translated: str,
        answer: str,
        response: str = "",
        is_correct: Optional[bool] = None,
    ):
        self.id = qid
        self.number = number
        self.question_text = text
        self.translated_question = translated
        self.answer = answer
        self.response = response
        self.is_correct = is_correct


MOCK_QUESTIONS = [
    MockQuestion(
        1,
        1,
        "What type of room did the guest initially request?",
        "Que tipo de quarto o hóspede solicitou inicialmente?",
        "A standard room",
        "Um quarto standard",
        True,
    ),
    MockQuestion(
        2,
        2,
        "What view options were offered?",
        "Quais opções de vista foram oferecidas?",
        "City views or garden views",
        "Vista da cidade ou vista do jardim",
        True,
    ),
    MockQuestion(
        3,
        3,
        "What is the nightly rate mentioned?",
        "Qual é a tarifa por noite mencionada?",
        "€120 per night",
        "€120 por noite",
        True,
    ),
    MockQuestion(
        4,
        4,
        "What is included in the room rate?",
        "O que está incluído na tarifa do quarto?",
        "Breakfast",
        "Pequeno-almoço",
        False,
    ),
    MockQuestion(
        5,
        5,
        "How much extra does late checkout cost?",
        "Quanto custa o checkout tardio?",
        "€30",
        "€30",
        True,
    ),
]


@app.route("/admin/preview")
@login_required
def admin_ui_preview_index():
    """Landing page for admin to choose which UI to preview."""
    if current_user.role != "admin":
        abort(403)

    previews = [
        {
            "name": "Sender Recording",
            "description": "The interface senders use to record their audio for a conversation.",
            "url": url_for("admin_preview_sender_recording"),
        },
        {
            "name": "Sender Final Survey",
            "description": "The final usability survey senders complete after validation.",
            "url": url_for("admin_preview_sender_survey"),
        },
        {
            "name": "Receiver Annotation",
            "description": "The interface receivers use to answer questions based on translated text.",
            "url": url_for("admin_preview_receiver"),
        },
        {
            "name": "Validator Annotation",
            "description": "The interface validators use to review translations and verify answers.",
            "url": url_for("admin_preview_validator"),
        },
    ]

    return render_template("admin_ui_preview_index.html", previews=previews)


@app.route("/admin/preview/sender-recording")
@login_required
def admin_preview_sender_recording():
    """Preview the sender recording annotation UI with mock data."""
    if current_user.role != "admin":
        abort(403)

    mock_conversation = MockConversation()
    mock_entry = MockEntry()
    mock_progress = {"percent": 40, "completed": 2, "total": 5}

    return render_template(
        "sender_annotation.html",
        entry=mock_entry,
        conversation=mock_conversation,
        recording=None,
        next_entry=None,
        progress=mock_progress,
        is_admin_preview=True,
    )


@app.route("/admin/preview/sender-survey")
@login_required
def admin_preview_sender_survey():
    """Preview the sender final survey UI with mock data."""
    if current_user.role != "admin":
        abort(403)

    mock_conversation = MockConversation()

    question_items = [
        {
            "id": q.id,
            "number": q.number,
            "text": q.translated_question,
            "validator_correct": q.is_correct,
        }
        for q in MOCK_QUESTIONS
    ]

    mock_progress = {"percent": 80, "completed": 4, "total": 5}

    return render_template(
        "final_survey.html",
        entry=MockEntry(),
        conversation=mock_conversation,
        recording_url=None,
        question_items=question_items,
        quality_score=5,
        selected_score=None,
        progress=mock_progress,
        is_admin_preview=True,
    )


@app.route("/admin/preview/receiver")
@login_required
def admin_preview_receiver():
    """Preview the receiver annotation UI with mock data."""
    if current_user.role != "admin":
        abort(403)

    mock_conversation = MockConversation()
    mock_entry = MockEntry()

    question_items = [
        {
            "id": q.id,
            "number": q.number,
            "translated_question": q.translated_question,
            "response": q.response,
        }
        for q in MOCK_QUESTIONS
    ]

    mock_progress = {"percent": 60, "completed": 3, "total": 5}

    translation_text = """Bom dia! Gostaria de reservar um quarto para duas noites, por favor.
Claro! Temos várias opções disponíveis. Prefere um quarto standard ou uma suite?
Um quarto standard seria óptimo. Tem vista?
Sim, temos quartos com vista para a cidade ou para o jardim. A vista para o jardim é ligeiramente mais silenciosa.
Fico com a vista para o jardim, por favor. Qual é a tarifa por noite?
Seria €120 por noite, incluindo pequeno-almoço. Deseja que prossiga com a reserva?
Sim, por favor. Posso também solicitar um checkout tardio no segundo dia?
Certamente! O checkout tardio até às 14h está disponível por um adicional de €30. Devo adicionar?
Sim, seria perfeito. Obrigado pela ajuda!"""

    translation_payload = {
        "translated_text": translation_text,
        "metadata": {"model": "demo-translation-model", "timestamp": "2025-01-01"},
    }

    receiver_locale = get_receiver_locale(DEFAULT_RECEIVER_UI_LOCALE)
    receiver_locale_strings: dict[str, str] = (
        receiver_locale.to_dict() if receiver_locale else {}
    )

    return render_template(
        "receiver_annotation.html",
        entry=mock_entry,
        conversation=mock_conversation,
        question_items=question_items,
        has_responses=True,
        is_complete=False,
        is_locked=False,
        next_entry=None,
        progress=mock_progress,
        translation_text=translation_text,
        translation_payload=translation_payload,
        receiver_locale=receiver_locale,
        receiver_locale_strings=receiver_locale_strings,
        initial_status_message="[Admin Preview] This is a demonstration of the receiver interface.",
        save_button_label="Save responses",
        next_button_label="Back to assignments",
        is_admin_preview=True,
    )


@app.route("/admin/preview/validator")
@login_required
def admin_preview_validator():
    """Preview the validator annotation UI with mock data."""
    if current_user.role != "admin":
        abort(403)

    mock_conversation = MockConversation()
    mock_entry = MockEntry()

    question_rows = [
        {
            "id": q.id,
            "number": q.number,
            "question_text": q.question_text,
            "translated_question": q.translated_question,
            "reference_answer": q.answer,
            "receiver_response": q.response,
            "is_correct": q.is_correct,
        }
        for q in MOCK_QUESTIONS
    ]

    mock_progress = {"percent": 50, "completed": 5, "total": 10}

    translation_text = """Bom dia! Gostaria de reservar um quarto para duas noites, por favor.

Claro! Temos várias opções disponíveis. Prefere um quarto standard ou uma suite?

Um quarto standard seria óptimo. Tem vista?

Sim, temos quartos com vista para a cidade ou para o jardim. A vista para o jardim é ligeiramente mais silenciosa.

Fico com a vista para o jardim, por favor. Qual é a tarifa por noite?

Seria €120 por noite, incluindo pequeno-almoço. Deseja que prossiga com a reserva?

Sim, por favor. Posso também solicitar um checkout tardio no segundo dia?

Certamente! O checkout tardio até às 14h está disponível por um adicional de €30. Devo adicionar?

Sim, seria perfeito. Obrigado pela ajuda!"""

    return render_template(
        "validator_annotation.html",
        entry=mock_entry,
        conversation=mock_conversation,
        translation_text=translation_text,
        translation_metadata={"model": "demo-translation-model"},
        question_rows=question_rows,
        translation_score=5,
        validator_review=None,
        is_locked=False,
        recording_url=None,
        next_entry=None,
        progress=mock_progress,
        is_admin_preview=True,
    )


@app.route("/logout")
@login_required
def logout():
    user_id = current_user.unique_id if current_user.is_authenticated else "unknown"
    logger.info(f"User {user_id} logged out")
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    """Return the WSGI application.

    Note: this module historically used a single global Flask app. For tests (and
    some embedding scenarios), callers may pass config overrides. When overrides
    are provided, we also clear any cached SQLAlchemy engines so the new
    configuration takes effect.
    """
    if config_overrides:
        app.config.update(config_overrides)

        # This codebase uses a global Flask app initialized at import-time.
        # Flask-SQLAlchemy builds an Engine for the configured database URL and
        # does not automatically rebind if the URL changes later.
        #
        # For tests, we want to safely override the DB URL (e.g., sqlite:///:memory:)
        # without touching any on-disk development database.
        if app.config.get("TESTING") and (
            "SQLALCHEMY_DATABASE_URI" in config_overrides
            or "SQLALCHEMY_BINDS" in config_overrides
        ):
            try:
                with app.app_context():
                    engines = db._app_engines.get(app)
                    if engines is not None:
                        engines.clear()

                        base_engine_options = dict(
                            getattr(db, "_engine_options", {}) or {}
                        )

                        default_uri = app.config.get("SQLALCHEMY_DATABASE_URI")
                        if default_uri:
                            default_options = dict(base_engine_options)
                            default_options["url"] = default_uri
                            engines[None] = db._make_engine(None, default_options, app)

                        binds = app.config.get("SQLALCHEMY_BINDS") or {}
                        for bind_key, bind_uri in binds.items():
                            bind_options = dict(base_engine_options)
                            bind_options["url"] = bind_uri
                            engines[bind_key] = db._make_engine(
                                bind_key, bind_options, app
                            )
            except Exception:
                logger.exception("Failed to rebuild SQLAlchemy engines")

    logger.info("Flask application created and ready")
    return app


def main() -> None:
    debug_flag = os.environ.get("FLASK_DEBUG", "0") in {"1", "true", "True"}
    port = int(os.environ.get("PORT", 5000))
    logger.info(
        f"Starting Flask development server on port {port} (debug={debug_flag})"
    )
    create_app().run(debug=debug_flag, port=port)


if __name__ == "__main__":
    main()
