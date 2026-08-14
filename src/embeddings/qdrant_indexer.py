import logging
import threading

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

BATCH_SIZE = 100

# Per-path cached clients for READ operations (search / collection info).
# Opening the embedded Qdrant storage on every request is expensive and, if
# an exception fires before close(), leaks file handles — on Windows those
# handles lock the storage directory and block later rebuilds. A per-path
# singleton keeps exactly one handle open per storage for the process
# lifetime. Write paths (create/upsert) still use short-lived clients and
# invalidate the cached one, because they replace collection data.
_clients: dict[str, QdrantClient] = {}
_clients_lock = threading.Lock()


def _get_read_client(path: str) -> QdrantClient:
    """Return the cached read client for a storage path (lazily opened)."""
    with _clients_lock:
        client = _clients.get(path)
        if client is None:
            client = QdrantClient(path=path)
            _clients[path] = client
            logger.debug("Opened cached Qdrant client for %s", path)
        return client


def _invalidate_read_client(path: str) -> None:
    """Drop and close the cached client after the storage was rewritten."""
    with _clients_lock:
        client = _clients.pop(path, None)
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


class QdrantIndexer:

    def create_collection(self, path: str, collection_name: str, vector_size: int):
        client = QdrantClient(path=path)
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        finally:
            client.close()
        _invalidate_read_client(path)

    def upsert_chunks(
        self,
        path: str,
        collection_name: str,
        ids: list,
        vectors: list,
        payloads: list,
    ):
        client = QdrantClient(path=path)
        try:
            points = [
                PointStruct(id=id_, vector=vec, payload=payload)
                for id_, vec, payload in zip(ids, vectors, payloads)
            ]
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i : i + BATCH_SIZE]
                client.upsert(collection_name=collection_name, points=batch)
        finally:
            client.close()
        _invalidate_read_client(path)

    def search(
        self, path: str, collection_name: str, query_vector: list, limit: int = 3
    ) -> list[dict]:
        client = _get_read_client(path)
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in response.points
        ]

    def get_collection_info(self, path: str, collection_name: str) -> dict:
        client = _get_read_client(path)
        info = client.get_collection(collection_name=collection_name)
        return {
            "points_count": info.points_count,
            "status": info.status,
        }
