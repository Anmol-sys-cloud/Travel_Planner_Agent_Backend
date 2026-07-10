import json
import os
import re
# pyrefly: ignore [missing-import]
from filelock import FileLock
from typing import Optional, List


class JsonStore:
    GENERIC_TITLES = {"new chat", "untitled", ""}

    def __init__(self, path: str):
        self.path = path
        self.lock_path = path + ".lock"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({"chats": []})
        self._cleanup_empty_chats()
        self._cleanup_generic_titles()

    def _should_keep_chat(self, chat: dict) -> bool:
        if chat.get("messages"):
            return True
        if not self._is_generic_title(chat.get("title")):
            return True
        return False

    def _cleanup_empty_chats(self):
        data = self._read()
        chats = [c for c in data.get("chats", []) if self._should_keep_chat(c)]
        if len(chats) != len(data.get("chats", [])):
            data["chats"] = chats
            self._write(data)

    def _is_generic_title(self, title: Optional[str]) -> bool:
        return not title or title.strip().lower() in self.GENERIC_TITLES

    def _make_title_from_text(self, text: str) -> Optional[str]:
        source = text.strip()
        if not source:
            return None

        source = re.sub(r"[^A-Za-z0-9 ]+", " ", source)
        source = re.sub(
            r"\b(plan|please|a|an|the|to|for|in|make|me|help|need|travel|trip|book|plan)\b",
            " ",
            source,
            flags=re.IGNORECASE,
        )
        source = re.sub(r"\s+", " ", source).strip()
        if not source:
            return None

        if len(source) > 40:
            source = source[:40].rsplit(" ", 1)[0]
        title = source.title()
        if not re.search(r"\b(Trip|Plan|Itinerary)\b", title, flags=re.IGNORECASE):
            title = f"{title} Plan"
        return title

        # Added till here..

    def _make_chat_title(self, chat: dict) -> Optional[str]:
        messages = chat.get("messages", []) or []
        if not messages:
            return None
        first_user = next((m for m in messages if m.get("role") == "user" and m.get("content")), None)
        source_text = first_user.get("content") if first_user else messages[0].get("content", "")
        if not source_text:
            return None
        return self._make_title_from_text(source_text)

    def _normalize_chat(self, chat: dict) -> bool:
        if self._is_generic_title(chat.get("title")) and chat.get("messages"):
            new_title = self._make_chat_title(chat)
            if new_title and not self._is_generic_title(new_title):
                chat["title"] = new_title
                return True
        return False

    def _cleanup_generic_titles(self):
        data = self._read()
        chats = data.get("chats", [])
        updated = False
        for chat in chats:
            if self._normalize_chat(chat):
                updated = True
        if updated:
            data["chats"] = chats
            self._write(data)

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

    def list_chats(self, user_id: Optional[str] = None) -> List[dict]:
        data = self._read()
        chats = [c for c in data.get("chats", []) if self._should_keep_chat(c)]
        if len(chats) != len(data.get("chats", [])):
            data["chats"] = chats
            self._write(data)
        if user_id is None:
            return chats
        return [c for c in chats if c.get("user_id") == user_id]

    def get_chat(self, chat_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        chats = self.list_chats(user_id=user_id)
        for c in chats:
            if c.get("id") == chat_id:
                return c
        return None

    def save_chat(self, chat: dict, user_id: Optional[str] = None):
        data = self._read()
        chats = data.get("chats", [])
        if user_id is not None:
            chat["user_id"] = user_id
        chats = [c for c in chats if c.get("id") != chat.get("id")]
        chats.insert(0, chat)
        data["chats"] = chats
        self._write(data)

    def delete_chat(self, chat_id: str, user_id: Optional[str] = None) -> bool:
        data = self._read()
        chats = data.get("chats", [])
        new = [c for c in chats if c.get("id") != chat_id or (user_id is not None and c.get("user_id") != user_id)]
        if len(new) == len(chats):
            return False
        data["chats"] = new
        self._write(data)
        return True

    def append_message(self, chat_id: str, message: dict, user_id: Optional[str] = None):
        data = self._read()
        chats = data.get("chats", [])
        for c in chats:
            if c.get("id") == chat_id and (user_id is None or c.get("user_id") == user_id):
                c.setdefault("messages", []).append(message)
                c["updated_at"] = message.get("ts")
                break
        else:
            raise KeyError("chat_not_found")
        data["chats"] = chats
        self._write(data)

    def update_chat_title(self, chat_id: str, title: str, user_id: Optional[str] = None):
        data = self._read()
        chats = data.get("chats", [])
        updated = False
        for c in chats:
            if c.get("id") == chat_id and (user_id is None or c.get("user_id") == user_id):
                c["title"] = title
                updated = True
                break
        if not updated:
            raise KeyError("chat_not_found")
        data["chats"] = chats
        self._write(data)

    def search_chats(self, q: str, user_id: Optional[str] = None) -> List[dict]:
        ql = q.lower()
        results = []
        for c in self.list_chats(user_id=user_id):
            if ql in (c.get("title") or "").lower():
                results.append(c)
                continue
            for m in c.get("messages", []):
                if ql in (m.get("content") or "").lower():
                    results.append(c)
                    break
        return results
