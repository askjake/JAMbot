#!/usr/bin/env python3

import asyncio
import sys
import yaml
from datetime import date
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import get_db_session_ctxmgr
from app.releases.repository import ReleaseRepository
from app.releases.schemas import ReleaseCreate


async def add_release(title: str, changes: list[str], release_date: date = None):
    """Add a new release to the database"""
    if release_date is None:
        release_date = date.today()

    repo = ReleaseRepository()

    async with get_db_session_ctxmgr() as db:
        # Check if a release already exists for this date
        existing_releases = await repo.list_releases(db, date=release_date)

        if existing_releases:
            existing_release = existing_releases[0]
            print(f"Release already exists for {release_date}: {existing_release.title} ({existing_release.release_id})")
            print(f"   Existing changes: {existing_release.changes}")
            return None

        # Create new release if no existing one found
        release_data = ReleaseCreate(title=title, date=release_date, changes=changes)
        release = await repo.create_one(db, obj_in=release_data)
        print(f"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Release created: {release.title} ({release.release_id})")
        return release


async def main():
    if len(sys.argv) != 2:
        print("Usage: python add_release_doc.py <yaml_file>")
        sys.exit(1)

    yaml_file = Path(sys.argv[1])
    if not yaml_file.exists():
        print(f"Error: {yaml_file} not found")
        sys.exit(1)

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    release_date = date.fromisoformat(data["date"]) if data.get("date") else None
    await add_release(data["title"], data["changes"], release_date)


if __name__ == "__main__":
    asyncio.run(main())
