#!/usr/bin/env python3
"""Database management helper: merge of reset_db.py and init_db.py.

Usage:
  python scripts/manage_db.py init
  python scripts/manage_db.py reset --yes
  python scripts/manage_db.py seed-admin
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure imports work regardless of the current working directory. We compute
# the repository and package directories relative to this script location
# then insert the project root on sys.path so `ouvia` can be imported.
_this_file = Path(__file__).resolve()
_ouvia_dir = _this_file.parents[1]
_project_root = _ouvia_dir.parent
sys.path.insert(0, str(_project_root))
from ouvia.app import User, app, db


def ensure_admin_user() -> None:
    admin_unique_id = "admin"
    admin_password = "adminouvia"

    existing_admin = User.query.filter_by(unique_id=admin_unique_id).first()
    if existing_admin:
        print("Admin user already exists. No changes applied.")
        return

    admin = User(unique_id=admin_unique_id, role="admin")
    admin.set_password(admin_password)
    admin.terms_accepted_at = datetime.now(timezone.utc)
    db.session.add(admin)
    db.session.commit()
    print("Admin credentials")
    print("-------------------")
    print(f"Unique ID: {admin_unique_id}")
    print(f"Password: {admin_password}")


def reset_database(yes: bool = False) -> None:
    if not yes:
        confirm = input(
            "This will DROP and re-create the database schema. Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Aborting.")
            return

    with app.app_context():
        db.drop_all()
        db.create_all()
        print("Database schema dropped and recreated successfully.")


def init_database() -> None:
    with app.app_context():
        db.create_all()
        ensure_admin_user()


def main() -> int:
    parser = argparse.ArgumentParser(description="Database management helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("reset", help="Drop and re-create the DB schema (destructive)")
    sub.add_parser("init", help="Create DB schema and seed the admin user")
    sub.add_parser("seed-admin", help="Seed the default admin user (non-destructive)")
    parser.add_argument("--yes", action="store_true", help="Answer yes to prompts")

    args = parser.parse_args()
    if args.command == "reset":
        reset_database(yes=args.yes)
    elif args.command == "init":
        init_database()
    elif args.command == "seed-admin":
        with app.app_context():
            ensure_admin_user()
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
