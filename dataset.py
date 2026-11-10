"""Compatibility entry point for refreshing the configured free data sources."""
from src.ingestion import _main


if __name__ == "__main__":
    raise SystemExit(_main())
