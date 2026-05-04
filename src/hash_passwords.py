"""hash_passwords.py — generate bcrypt hashes for the secrets TOML.

Plaintext stays in stdin (via getpass — no shell history, no screen echo).
Output is a TOML snippet you paste into Streamlit Cloud → App settings →
Secrets. Plaintext is never written to disk.

Run:
    cd src
    python hash_passwords.py
"""
import getpass
import secrets as pysecrets

import streamlit_authenticator as stauth


def main() -> None:
    print("Hash generator for Streamlit Cloud secrets.")
    print("Plaintext stays in stdin; press Ctrl+C to abort.\n")

    creds = {"usernames": {}}

    while True:
        username = input("Username (blank to finish): ").strip()
        if not username:
            break
        first = input("  First name: ").strip()
        last  = input("  Last name:  ").strip()
        email = input("  Email:      ").strip()
        pwd     = getpass.getpass("  Password:   ")
        confirm = getpass.getpass("  Confirm:    ")
        if pwd != confirm:
            print("  Passwords don't match — skipping user.\n")
            continue
        if len(pwd) < 8:
            print("  Password is shorter than 8 characters — skipping user.\n")
            continue
        creds["usernames"][username] = {
            "first_name": first,
            "last_name":  last,
            "email":      email,
            "password":   pwd,
        }
        print()

    if not creds["usernames"]:
        print("No users entered — exiting.")
        return

    # streamlit-authenticator 0.4.x: Hasher.hash_passwords mutates in-place,
    # replacing each plaintext "password" with its bcrypt hash.
    stauth.Hasher.hash_passwords(creds)

    cookie_key = pysecrets.token_hex(32)

    print("=" * 64)
    print("Paste this into Streamlit Cloud → App settings → Secrets:")
    print("(REAL VALUES NEVER GO IN GIT)")
    print("=" * 64)
    print()
    print("[app]")
    print("cloud_mode = true")
    print()
    for uname, u in creds["usernames"].items():
        print(f"[auth.users.{uname}]")
        print(f'first_name    = "{u["first_name"]}"')
        print(f'last_name     = "{u["last_name"]}"')
        print(f'email         = "{u["email"]}"')
        print(f'password_hash = "{u["password"]}"')
        print()
    print("[auth.cookie]")
    print('name        = "paper_trader_dashboard"')
    print(f'key         = "{cookie_key}"')
    print('expiry_days = 7')
    print()
    print("[r2]")
    print('endpoint_url      = "<paste R2_ENDPOINT_URL>"')
    print('access_key_id     = "<paste R2_ACCESS_KEY_ID>"')
    print('secret_access_key = "<paste R2_SECRET_ACCESS_KEY>"')
    print('bucket_name       = "<paste R2_BUCKET_NAME>"')
    print()


if __name__ == "__main__":
    main()
