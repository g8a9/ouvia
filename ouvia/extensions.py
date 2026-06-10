from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# Shared extension instances. Initialize in the application factory.
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
