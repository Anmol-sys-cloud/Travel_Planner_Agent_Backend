import json
import os
import datetime
# pyrefly: ignore [missing-import]
from filelock import FileLock
from typing import Optional, List


class UserStore:
    def __init__(self, path: str):
        self.path = path
        self.lock_path = path + ".lock"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({"users": []})

    def _read(self) -> dict:
        lock = FileLock(self.lock_path)
        with lock:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write(self, data: dict):
        lock = FileLock(self.lock_path)
        with lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def list_users(self) -> List[dict]:
        data = self._read()
        return data.get("users", [])

    def get_user(self, user_id: str) -> Optional[dict]:
        for user in self.list_users():
            if user.get("id") == user_id:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        for user in self.list_users():
            if user.get("email", "").lower() == email.lower():
                return user
        return None

    def save_user(self, user: dict) -> dict:
        data = self._read()
        users = data.get("users", [])
        now = datetime.datetime.utcnow().isoformat()
        user["created_at"] = user.get("created_at", now)
        user["updated_at"] = now
        user.setdefault("profile", {"phone": "", "bio": ""})
        users = [u for u in users if u.get("id") != user.get("id")]
        users.insert(0, user)
        data["users"] = users
        self._write(data)
        return user

    def update_user(self, user_id: str, updates: dict) -> Optional[dict]:
        data = self._read()
        users = data.get("users", [])
        updated = False
        for user in users:
            if user.get("id") == user_id:
                user.update({k: v for k, v in updates.items() if v is not None})
                user["updated_at"] = datetime.datetime.utcnow().isoformat()
                updated = True
                break
        if not updated:
            return None
        data["users"] = users
        self._write(data)
        return next((u for u in users if u.get("id") == user_id), None)

    def delete_user(self, user_id: str) -> bool:
        data = self._read()
        users = data.get("users", [])
        new_users = [u for u in users if u.get("id") != user_id]
        if len(new_users) == len(users):
            return False
        data["users"] = new_users
        self._write(data)
        return True
