import json
from os import getenv

import firebase_admin
import requests
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from firebase_admin import auth, credentials
from requests.exceptions import HTTPError

from models.user_model import LoginSchema, SignUpSchema

users_router = APIRouter()


@users_router.post("/signup")
async def signup(user_data: SignUpSchema):
    email = user_data.email
    password = user_data.password

    auth.tenant_id = "aggreko"

    try:
        user = auth.create_user(email=email, password=password)

        return JSONResponse(
            content={
                "message": f"User account created successfuly for user {user.uid}"
            },
            status_code=201,
        )
    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=400, detail=f"Account already created for the email {email}"
        )


@users_router.post("/login")
def sign_in_with_email_and_password(user_data: SignUpSchema):
    email = user_data.email
    password = user_data.password

    return_secure_token = True

    # payload = json.dumps({"email":email, "password":password, "return_secure_token": "True"})
    data = json.dumps(
        {
            "email": email,
            "password": password,
            "return_secure_token": return_secure_token,
        }
    )

    FIREBASE_WEB_API_KEY = getenv("FIREBASE_WEB_API_KEY", None)

    request_ref = f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key={FIREBASE_WEB_API_KEY}"

    request_object = requests.post(
        request_ref,
        headers={"content-type": "application/json; charset=UTF-8"},
        data=data,
    )

    try:
        request_object.raise_for_status()
    except HTTPError as e:
        # raise detailed error message
        # TODO: Check if we get a { "error" : "Permission denied." } and handle automatically
        raise HTTPError(e, request_object.text)

    return request_object.json()["idToken"]


@users_router.post("/ping")
async def validate_token(request: Request):
    headers = request.headers
    jwt_token = headers.get("Authorization")

    if not jwt_token:
        raise HTTPException(status_code=400, detail="TokenID must be provided")

    auth.tenant_id = "aggreko"

    try:
        claims = auth.verify_id_token(jwt_token)
        return claims
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=401, detail="Unauthorized")
