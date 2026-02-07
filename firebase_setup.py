"""firebase_setup.py

Helper to initialize the Firebase Admin SDK using the `FIREBASE_CREDENTIALS`
value from `.env`. It can optionally write the parsed service account JSON
to a file for manual inspection.

Usage examples:
  python firebase_setup.py            # initialize and test Firestore access
  python firebase_setup.py --write key.json  # also write the service key to key.json

IMPORTANT: Do not commit your service account JSON to version control.
"""
import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

def load_creds_from_env():
    firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    if not firebase_creds_json:
        raise RuntimeError('FIREBASE_CREDENTIALS not found in environment.')

    # Strip surrounding quotes if present
    if (firebase_creds_json.startswith("'") and firebase_creds_json.endswith("'")) or (
        firebase_creds_json.startswith('"') and firebase_creds_json.endswith('"')
    ):
        firebase_creds_json = firebase_creds_json[1:-1]

    # Try parsing JSON; if escaped newlines exist, unescape them and retry.
    try:
        creds = json.loads(firebase_creds_json)
    except Exception:
        firebase_creds_json = firebase_creds_json.replace('\\n', '\n')
        creds = json.loads(firebase_creds_json)

    return creds


def init_and_test_firestore(creds_dict):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'projectId': creds_dict.get('project_id')})
            print('Firebase initialized successfully.')
        else:
            print('Firebase app already initialized.')

        db = firestore.client()
        # Try a light read: list top-level collections (may require permissions)
        try:
            col_iter = db.collections()
            cols = [c.id for c in col_iter]
            print(f'Accessible top-level collections: {cols}')
        except Exception as e:
            print(f'Initialized Firebase but failed to list collections: {e}')

    except Exception as e:
        print(f'Failed to initialize Firebase Admin SDK: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', '-w', help='Write service account JSON to this path')
    args = parser.parse_args()

    try:
        creds = load_creds_from_env()
    except Exception as e:
        print(f'Error loading credentials from environment: {e}')
        return

    if args.write:
        try:
            with open(args.write, 'w', encoding='utf-8') as f:
                json.dump(creds, f, indent=2)
            print(f'Wrote service account JSON to {args.write} (remove after use)')
        except Exception as e:
            print(f'Failed to write service account JSON to disk: {e}')

    init_and_test_firestore(creds)


if __name__ == '__main__':
    main()
