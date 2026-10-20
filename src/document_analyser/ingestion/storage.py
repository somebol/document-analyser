"""Storage keys, rather than local paths, are persisted in PostgreSQL."""

import shutil
from pathlib import Path
from typing import BinaryIO, Protocol

from document_analyser.config import get_settings


class Storage(Protocol):
    # Persist a new immutable object from a binary stream.
    def put(self, key: str, source: BinaryIO) -> None: ...

    # Materialise an object at a worker-controlled temporary path.
    def materialize(self, key: str, destination: Path) -> None: ...

    # Remove an unregistered object after a failed upload transaction.
    def delete(self, key: str) -> None: ...


class LocalStorage:
    # Use a shared storage root for independently running API and worker processes.
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    # Reject traversal and symlink escapes even for internally generated keys.
    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Invalid storage key")
        return path

    # Exclusively create immutable objects and clean partial writes on failure.
    def put(self, key: str, source: BinaryIO) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            try:
                shutil.copyfileobj(source, output)
            except BaseException:
                target.unlink(missing_ok=True)
                raise

    # Copy locally stored objects to a caller-owned working directory.
    def materialize(self, key: str, destination: Path) -> None:
        shutil.copyfile(self.path(key), destination)

    # Delete an orphan without exposing arbitrary filesystem paths.
    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)


# Construct the storage adapter using shared application settings.
def get_storage() -> Storage:
    return LocalStorage(get_settings().document_storage_dir)
