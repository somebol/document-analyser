from functools import partial

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from document_analyser import config
from document_analyser.ingestion.storage import LocalStorage
from document_analyser.persistence.models import Base


# Exercise relational transactions with foreign keys, without downloading models.
@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


# Keep all synthetic originals and artifacts isolated from personal documents.
@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "objects")


# Keep developer environment files and cached settings out of every test.
@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in config.Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(config, "Settings", partial(config.Settings, _env_file=None))
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
