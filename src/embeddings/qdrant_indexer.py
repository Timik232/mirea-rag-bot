from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


BATCH_SIZE = 100


class QdrantIndexer:

    def create_collection(self, path: str, collection_name: str, vector_size: int):
        client = QdrantClient(path=path)
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        client.close()

    def upsert_chunks(
        self,
        path: str,
        collection_name: str,
        ids: list,
        vectors: list,
        payloads: list,
    ):
        client = QdrantClient(path=path)
        points = [
            PointStruct(id=id_, vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            client.upsert(collection_name=collection_name, points=batch)
        client.close()

    def search(
        self, path: str, collection_name: str, query_vector: list, limit: int = 3
    ) -> list[dict]:
        client = QdrantClient(path=path)
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        client.close()
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in response.points
        ]

    def get_collection_info(self, path: str, collection_name: str) -> dict:
        client = QdrantClient(path=path)
        info = client.get_collection(collection_name=collection_name)
        client.close()
        return {
            "points_count": info.points_count,
            "status": info.status,
        }
