import os
from pathlib import Path


class Config:
    # App config
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")

    # Default DB path: absolute path to study_portal/instance/app.db
    _default_db_path = os.environ.get(
        "SQLALCHEMY_DATABASE_URI",
        "sqlite:///"
        + os.path.abspath(
            os.path.join(os.path.dirname(__file__), "instance", "app.db")
        ),
    )
    SQLALCHEMY_DATABASE_URI = _default_db_path
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Files / directories
    RECORDINGS_DIR = os.environ.get(
        "RECORDINGS_DIR",
        os.path.join(os.path.dirname(__file__), "recordings"),
    )
    TRANSLATIONS_DIR = os.environ.get(
        "TRANSLATIONS_DIR",
        os.path.join(os.path.dirname(__file__), "translations"),
    )

    # Signup keys (role -> key) - override with env var if necessary
    ROLE_SIGNUP_KEYS = {
        "sender": os.environ.get("ROLE_SIGNUP_KEY_SENDER", "8f4c2a7b10d3"),
        "final_sender": os.environ.get("ROLE_SIGNUP_KEY_FINAL_SENDER", "d7e9f3a1b5c2"),
        "receiver": os.environ.get("ROLE_SIGNUP_KEY_RECEIVER", "5b19e3ad72c8"),
        "validator": os.environ.get("ROLE_SIGNUP_KEY_VALIDATOR", "c0ffee12ab34"),
    }

    # Sender demographics
    SENDER_GENDER_OPTIONS = ["man", "woman", "non-binary"]
    SENDER_GENDER_LABELS = {
        "man": "Man (including Trans Male / Trans Man)",
        "woman": "Woman (including Trans Female / Trans Woman)",
        "non-binary": "Non-binary (would like to give more detail)",
    }
    # Whether the participant is trans (i.e., their current gender differs from
    # the one they were assigned at birth).
    SENDER_TRANS_OPTIONS = ["yes", "no", "rather-not-say"]
    SENDER_TRANS_LABELS = {
        "yes": "Yes",
        "no": "No",
        "rather-not-say": "Rather not say",
    }

    # Backward-compatible aliases (older deployments referenced *_SEX_* keys).
    SENDER_SEX_OPTIONS = SENDER_TRANS_OPTIONS
    SENDER_SEX_LABELS = SENDER_TRANS_LABELS

    # Default assignment batch sizes (can be overridden via env)
    DEFAULT_SENDER_ASSIGNMENTS = int(os.environ.get("DEFAULT_SENDER_ASSIGNMENTS", "10"))
    DEFAULT_RECEIVER_ASSIGNMENTS = int(
        os.environ.get("DEFAULT_RECEIVER_ASSIGNMENTS", "3")
    )
    DEFAULT_VALIDATOR_ASSIGNMENTS = int(
        os.environ.get("DEFAULT_VALIDATOR_ASSIGNMENTS", "3")
    )
    DEFAULT_FINAL_SENDER_ASSIGNMENTS = int(
        os.environ.get("DEFAULT_FINAL_SENDER_ASSIGNMENTS", "10")
    )

    # UI theme colours - used to provide CSS variables in templates
    UI_PRIMARY_COLOR = os.environ.get("UI_PRIMARY_COLOR", "#698E6D")
    UI_SECONDARY_COLOR = os.environ.get("UI_SECONDARY_COLOR", "#F1972F")
    UI_LIGHT_BG = os.environ.get("UI_LIGHT_BG", "#f8f9fa")

    # Role-specific confirmation codes (all users of a role share the same code)
    ROLE_CONFIRMATION_CODES = {
        "sender": os.environ.get("CONFIRMATION_CODE_SENDER", "SENDER001"),
        "final_sender": os.environ.get("CONFIRMATION_CODE_FINAL_SENDER", "FSENDER001"),
        "receiver": os.environ.get("CONFIRMATION_CODE_RECEIVER", "RECEIVER001"),
        "validator": os.environ.get("CONFIRMATION_CODE_VALIDATOR", "VALIDATOR001"),
    }

    # Additional app settings (defaults)
    TERMS_ENFORCEMENT_SETTING_KEY = "require_terms_acceptance"
