from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from document_analyser.config import get_settings


# Share a connection pool within each API or worker process.
@lru_cache
def session_factory() -> sessionmaker[Session]:
    settings = get_settings()
    engine = create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )
    return sessionmaker(engine, expire_on_commit=False)