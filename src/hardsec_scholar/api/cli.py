"""Development entry point for the local HardSec Scholar API."""

import uvicorn


def main() -> None:
    """Run the local API with source reload enabled."""
    uvicorn.run(
        "hardsec_scholar.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
