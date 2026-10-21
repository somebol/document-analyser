import os
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


# Wait for Qdrant before allowing application services to start.
def wait_for_qdrant() -> None:
    base_url = os.environ["QDRANT_URL"].rstrip("/")
    deadline = time.monotonic() + 120

    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/readyz", timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass

        time.sleep(2)

    raise RuntimeError("Qdrant did not become ready within 120 seconds.")


# Apply schema migrations before starting API and worker processes.
def main() -> None:
    wait_for_qdrant()
    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
    )


if __name__ == "__main__":
    main()