from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
import uuid
from typing import Any

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, collection_name: str | None = None) -> None:
        self.collection_name = collection_name or settings.qdrant_collection
        self._client = None
        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=settings.qdrant_url, timeout=5)
        except Exception:
            self._client = None
        self.fallback_path = settings.qdrant_storage_path
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def ready(self) -> bool:
        if self._client is None:
            return self.rest_ready()
        try:
            self._client.get_collections()
            return True
        except Exception:
            return self.rest_ready()

    def rest_ready(self) -> bool:
        try:
            self._request_json("GET", "/collections")
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            self.collection_info()
            return True
        except Exception:
            return False

    def collection_info(self) -> dict[str, Any]:
        return self._request_json("GET", f"/collections/{self.collection_name}")

    def point_count(self) -> int | None:
        try:
            return int(self.collection_info().get("result", {}).get("points_count", 0))
        except Exception:
            return None

    def vector_size(self) -> int | None:
        try:
            vectors = self.collection_info().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            if isinstance(vectors, dict) and "size" in vectors:
                return int(vectors["size"])
        except Exception:
            return None
        return None

    def ensure_collection(self, vector_size: int) -> None:
        if self._client is None:
            self._ensure_collection_rest(vector_size)
            return
        try:
            from qdrant_client.http.models import Distance, VectorParams

            names = [item.name for item in self._client.get_collections().collections]
            if self.collection_name not in names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception:
            logger.warning("qdrant_client ensure_collection failed; using REST fallback.", exc_info=True)
            self._client = None
            self._ensure_collection_rest(vector_size)

    def _ensure_collection_rest(self, vector_size: int) -> None:
        if self.collection_exists():
            return
        payload = {"vectors": {"size": vector_size, "distance": "Cosine"}}
        self._request_json("PUT", f"/collections/{self.collection_name}", payload)

    def upsert(self, vector: list[float], payload: dict[str, Any], point_id: str | None = None) -> str:
        point_id = point_id or str(uuid.uuid4())
        if self._client is not None:
            try:
                from qdrant_client.http.models import PointStruct

                self.ensure_collection(len(vector))
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
                return point_id
            except Exception:
                logger.warning("qdrant_client upsert failed; using REST fallback.", exc_info=True)
                self._client = None
        try:
            self.ensure_collection(len(vector))
            self._request_json(
                "PUT",
                f"/collections/{self.collection_name}/points?wait=true",
                {"points": [{"id": point_id, "vector": vector, "payload": payload}]},
            )
            return point_id
        except Exception:
            logger.warning("Qdrant REST upsert failed; writing to fallback vector file.", exc_info=True)
        with self.fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": point_id, "vector": vector, "payload": payload}) + "\n")
        return point_id

    def delete_by_file_path(self, file_path: str) -> int:
        return self.delete_by_payload("file_path", file_path)

    def delete_by_payload(self, key: str, value: str) -> int:
        deleted = 0
        if self._client is not None:
            try:
                from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])
                    ),
                    wait=True,
                )
                deleted = -1
            except Exception:
                logger.warning("qdrant_client delete failed; using REST fallback.", exc_info=True)
                self._client = None
        if deleted == 0:
            try:
                self._request_json(
                    "POST",
                    f"/collections/{self.collection_name}/points/delete?wait=true",
                    {"filter": {"must": [{"key": key, "match": {"value": value}}]}},
                )
                deleted = -1
            except Exception:
                logger.debug("Qdrant REST delete failed; updating fallback vector file if present.", exc_info=True)
        return self._delete_from_fallback(key, value) if self.fallback_path.exists() else deleted

    def search(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if self._client is not None:
            try:
                hits = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True,
                )
                return [
                    {"id": str(hit.id), "score": float(hit.score), "payload": dict(hit.payload or {})}
                    for hit in hits
                ]
            except Exception:
                logger.warning("qdrant_client search failed; using REST fallback.", exc_info=True)
                self._client = None
        try:
            raw = self.search_raw(vector, top_k)
            hits = raw.get("result", [])
            return [
                {"id": str(hit.get("id")), "score": float(hit.get("score", 0.0)), "payload": dict(hit.get("payload") or {})}
                for hit in hits
            ]
        except Exception:
            logger.warning("Qdrant REST search failed; using fallback vector file.", exc_info=True)
        rows = []
        if not self.fallback_path.exists():
            return []
        with self.fallback_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(
                    {
                        "id": item["id"],
                        "score": cosine_similarity(vector, item["vector"]),
                        "payload": item["payload"],
                    }
                )
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]

    def keyword_search(self, symbols: list[str], file_fragments: set[str], limit: int = 50) -> list[dict[str, Any]]:
        if not symbols and not file_fragments:
            return []
        rows = self._scroll_payload_rows(limit=max(limit * 4, 200))
        matches = []
        for row in rows:
            payload = row.get("payload", {})
            content = str(payload.get("content", "")).lower()
            symbol_name = str(payload.get("symbol_name") or "").lower()
            file_path = str(payload.get("file_path") or "").replace("\\", "/").lower()
            folder = str(payload.get("folder") or "").replace("\\", "/").lower()
            imports = " ".join(str(item) for item in payload.get("imports") or []).lower()
            called = " ".join(str(item) for item in payload.get("called_functions") or []).lower()
            search_blob = " ".join([content, symbol_name, file_path, folder, imports, called])
            symbol_match = any(symbol.lower() in search_blob for symbol in symbols)
            path_match = any(fragment and fragment in file_path for fragment in file_fragments)
            if symbol_match or path_match:
                matches.append({"id": row.get("id"), "score": 0.55, "payload": payload})
            if len(matches) >= limit:
                break
        return matches

    def scroll_payload_rows(self, limit: int = 2000) -> list[dict[str, Any]]:
        return self._scroll_payload_rows(limit=limit)

    def _scroll_payload_rows(self, limit: int = 200) -> list[dict[str, Any]]:
        if self._client is not None:
            try:
                points, _ = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                return [
                    {"id": str(point.id), "payload": dict(point.payload or {})}
                    for point in points
                ]
            except Exception:
                logger.debug("qdrant_client scroll failed; using REST/fallback.", exc_info=True)
                self._client = None
        try:
            rows: list[dict[str, Any]] = []
            offset: Any = None
            while len(rows) < limit:
                payload: dict[str, Any] = {
                    "limit": min(256, limit - len(rows)),
                    "with_payload": True,
                    "with_vector": False,
                }
                if offset is not None:
                    payload["offset"] = offset
                raw = self._request_json("POST", f"/collections/{self.collection_name}/points/scroll", payload)
                result = raw.get("result", {})
                points = result.get("points", [])
                rows.extend(
                    {"id": str(point.get("id")), "payload": dict(point.get("payload") or {})}
                    for point in points
                )
                offset = result.get("next_page_offset")
                if not offset or not points:
                    break
            return rows
        except Exception:
            pass
        rows = []
        if not self.fallback_path.exists():
            return rows
        with self.fallback_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append({"id": item.get("id"), "payload": item.get("payload", {})})
                if len(rows) >= limit:
                    break
        return rows

    def _delete_from_fallback(self, key: str, value: str) -> int:
        kept: list[dict[str, Any]] = []
        deleted = 0
        if not self.fallback_path.exists():
            return 0
        with self.fallback_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if str(item.get("payload", {}).get(key)) == value:
                    deleted += 1
                else:
                    kept.append(item)
        with self.fallback_path.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item) + "\n")
        return deleted

    def search_raw(self, vector: list[float], top_k: int = 5) -> dict[str, Any]:
        payload = {"vector": vector, "limit": top_k, "with_payload": True}
        try:
            return self._request_json("POST", f"/collections/{self.collection_name}/points/search", payload)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            query_payload = {"query": vector, "limit": top_k, "with_payload": True}
            return self._request_json("POST", f"/collections/{self.collection_name}/points/query", query_payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = settings.qdrant_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
