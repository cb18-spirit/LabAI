from auth import register_user, login_user

success, message = register_user(
    "Chethan",
    "test@test.com",
    "123456",
    "student"
)

print(success, message)

success, user = login_user(
    "test@test.com",
    "123456"
)

print(success)
print(user)