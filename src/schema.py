from pydantic import BaseModel, Field


class Say(BaseModel):
    message: str


class ImageRecord(BaseModel):
    image_id: str
    user_id: str
    record_id: str
    s3_bucket_key: str | None = None
    s3_bucket_name: str | None = None
    content_type: str
    message_id: str | None = None  # In case, SQS was used


class User(BaseModel):
    identifier: str
    display_name: str | None = None
    metadata: dict = Field(default_factory=dict)
