from pydantic import BaseModel


class SignUpSchema(BaseModel):
    email: str
    password: str

    class Config:
        schema_extra = {
            "example": {"email": "john@doe.com", "password": "mycypheredpassword"}
        }


class LoginSchema(BaseModel):
    email: str
    password: str

    class Config:
        schema_extra = {"example": {"email": "jane@doe.com", "password": "sameasjohn"}}
