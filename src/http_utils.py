# Shared HTTP helpers: in-memory TTL cache and GET-with-backoff for rate-limited upstreams

import logging
import random
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRY_AFTER = 10.0  

_client: Optional[httpx.Client] = None


def _shared_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=12.0)
    return _client


class TTLCache:
    def __init__(self, ttl_seconds: float, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: Dict[Any, Tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: Any, max_age: Optional[float] = None) -> Optional[Tuple[Any, float]]:
        limit = self.ttl if max_age is None else max_age
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            age = time.monotonic() - stored_at
            if age > limit:
                return None
            return value, age

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if key not in self._data and len(self._data) >= self.max_items:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest]
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def _retry_delay(attempt: int, resp: Optional[httpx.Response]) -> float:
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), _MAX_RETRY_AFTER)
            except ValueError:
                pass
    # 1s, 2s, 4s... capped at 8s, plus jitter
    return min(2 ** attempt, 8) + random.uniform(0, 0.5)


def get_with_retry(
    url: str,
    params: Dict[str, Any],
    client: Optional[httpx.Client] = None,
    attempts: int = 3,
    timeout: float = 12.0,
) -> Tuple[Optional[httpx.Response], Optional[str]]:
    last_resp: Optional[httpx.Response] = None
    last_exc: Optional[str] = None

    for attempt in range(attempts):
        resp: Optional[httpx.Response] = None
        try:
            http = client if client is not None else _shared_client()
            resp = http.get(url, params=params, timeout=timeout)
            last_resp, last_exc = resp, None
            if resp.status_code == 200:
                return resp, None
            if resp.status_code not in RETRYABLE_STATUS:
                return resp, None
        except httpx.RequestError as exc:
            last_resp, last_exc = None, str(exc)

        if attempt < attempts - 1:
            delay = _retry_delay(attempt, resp)
            logger.warning(
                "GET %s failed (%s). Retry %d/%d in %.1fs.",
                url,
                resp.status_code if resp is not None else last_exc,
                attempt + 1,
                attempts - 1,
                delay,
            )
            time.sleep(delay)

    return last_resp, last_exc