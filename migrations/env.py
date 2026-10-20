from alembic import context

from document_analyser.persistence.database import session_factory
from document_analyser.persistence.models import Base


# Run versioned migrations using the same configured database as the application.
def run_migrations() -> None:
    engine = session_factory().kw["bind"]
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
