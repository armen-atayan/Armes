from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CallRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    contact_name: str = Field(min_length=1, max_length=120)
    phone_number: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    task: str = Field(min_length=1, max_length=2000)
    details: str = Field(default="", max_length=4000)
    origin: Literal["web_demo"] = "web_demo"
    owner_channel: Literal["web"] = "web"
    result_channel: Literal["web"] = "web"
    demo_session_id: str | None = None

    @field_validator("task")
    @classmethod
    def task_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task must not be blank")
        return value


class CallDispatchResult(BaseModel):
    room_name: str
    call_id: str | None = None


class CallCreated(BaseModel):
    session_id: str
    room_name: str


class OwnerResponse(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    response: str = Field(min_length=1, max_length=2000)


class LiveInstruction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000)


class FollowUpRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    action: Literal["continue", "cancel"]
    instruction: str = Field(default="", max_length=2000)
