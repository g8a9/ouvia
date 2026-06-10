ROOT := $(shell pwd)
PY := python3
UV := uv
FLASK_APP := app:create_app
DB_URI := sqlite:////$(ROOT)/instance/app.db
PORT := 5001

.PHONY: install run dev prod migrate-init migrate migrate-upgrade migrate-upgrade-rev migrate-downgrade migrate-stamp db-backup db-init db-reset db-seed shell

install:
	$(UV) pip install -r requirements.txt

run:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask run --port $(PORT)

# Shortcut for local development with Flask debug/reload enabled. Use this
# when you want templates and code changes to reload automatically.
debug:
	FLASK_APP=$(FLASK_APP) FLASK_DEBUG=1 SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask run --port $(PORT)

dev: run

prod:
	SQLALCHEMY_DATABASE_URI=$(DB_URI) gunicorn --workers 4 --bind 0.0.0.0:$(PORT) "ouvia.app:create_app()"

migrate-init:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db init --directory migrations

migrate:
	@echo "Creating migration: $(MSG)"
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db migrate -m "$(MSG)" --directory migrations

migrate-upgrade:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db upgrade --directory migrations

migrate-upgrade-rev:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db upgrade $(REV) --directory migrations

migrate-downgrade:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db downgrade $(REV) --directory migrations

migrate-stamp:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask db stamp $(REV) --directory migrations

db-backup:
	cp instance/app.db instance/app.db.bak

db-init:
	$(PY) scripts/manage_db.py init

db-reset:
	$(PY) scripts/manage_db.py reset

db-seed:
	$(PY) scripts/manage_db.py seed-admin

shell:
	FLASK_APP=$(FLASK_APP) SQLALCHEMY_DATABASE_URI=$(DB_URI) $(PY) -m flask shell
