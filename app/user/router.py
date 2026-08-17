from fastapi import APIRouter

from .schemas import User
from app.dependencies import UserEmailDep, DBSessionDep
from .service import get_last_release_date

router = APIRouter()

# whoami
@router.get("/whoami", tags=["user"])
async def get_user(db: DBSessionDep, email: UserEmailDep) -> User:
    '''Simply parse request header and respond user information.'''
    last_release_date = await get_last_release_date(db, user_email=email)
    return User(
        email=email,
        first_name=email.split('@')[0].split('.')[0].capitalize(),
        last_name=email.split('@')[0].split('.')[1].capitalize(),
        last_release_date=last_release_date
    )

