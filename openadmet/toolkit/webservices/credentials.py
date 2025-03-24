from pydantic import Field
from pydantic_settings import BaseSettings


class S3Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = Field(description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(description="AWS secret access key")
