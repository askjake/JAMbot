from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SNOWIncident(BaseModel):
    sys_id: str = ""
    number: str = ""
    short_description: str = ""
    state: str = ""
    priority: str = ""
    assigned_to: str = ""
    opened_at: str = ""
    updated_at: str = ""


class SNOWChange(BaseModel):
    sys_id: str = ""
    number: str = ""
    short_description: str = ""
    state: str = ""
    type: str = ""
    start_date: str = ""
    end_date: str = ""


class SNOWUser(BaseModel):
    sys_id: str = ""
    name: str = ""
    email: str = ""
    department: str = ""
    title: str = ""
