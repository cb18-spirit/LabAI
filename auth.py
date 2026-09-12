import bcrypt
from database import users_collection


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )


def verify_password(password, hashed_password):
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed_password
    )


def register_user(name, email, password, role):

    existing = users_collection.find_one(
        {"email": email}
    )

    if existing:
        return False, "User already exists"

    users_collection.insert_one(
        {
            "name": name,
            "email": email,
            "password": hash_password(password),
            "role": role
        }
    )

    return True, "Registration successful"


def login_user(email, password):

    user = users_collection.find_one(
        {"email": email}
    )

    if not user:
        return False, None

    if verify_password(
        password,
        user["password"]
    ):
        return True, user

    return False, None