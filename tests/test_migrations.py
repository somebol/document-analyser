from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from document_analyser.config import get_settings
from document_analyser.persistence.database import session_factory


# Verify the frozen migration can upgrade, match metadata and roll back cleanly.
def test_migration_roundtrip(tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    session_factory.cache_clear()
    try:
        config = Config("alembic.ini")
        command.upgrade(config, "head")
        command.check(config)
        engine = create_engine(url)
        assert "ner_mentions" in inspect(engine).get_table_names()
        command.downgrade(config, "base")
        assert inspect(engine).get_table_names() == ["alembic_version"]
        engine.dispose()
    finally:
        session_factory().kw["bind"].dispose()
        session_factory.cache_clear()
