import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.qdrant_store import QdrantStore
from backend.config.settings import ROOT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_qdrant():
    store = QdrantStore()
    
    if not store.wait_for_qdrant(retries=3, base_delay=1.0):
        logger.error("Failed to reach Qdrant. Please ensure the Docker container is running.")
        logger.error("Run: docker compose up -d qdrant")
        sys.exit(1)
    
    # 1. Delete the collection entirely
    if store._client is not None:
        try:
            logger.info(f"Attempting to delete collection: {store.collection_name}")
            store._client.delete_collection(collection_name=store.collection_name)
            logger.info("Collection deleted successfully via qdrant_client.")
        except Exception as e:
            logger.warning(f"Failed to delete collection via client: {e}")
            try:
                store._request_json("DELETE", f"/collections/{store.collection_name}")
                logger.info("Collection deleted via REST API.")
            except Exception as e2:
                logger.warning(f"Failed to delete collection via REST: {e2}")
    
    # 2. Re-create the collection with explicit vector size (768 for nomic-embed-text)
    # The default size for nomic-embed-text is 768. 
    # Ensuring collection explicitly avoids auto-creation issues.
    logger.info("Re-creating collection...")
    store.ensure_collection(vector_size=768)
    
    # 3. Apply Payload Index to prevent LiteralOutOfBounds on delete_by_filter
    if store._client is not None:
        try:
            from qdrant_client.http.models import PayloadSchemaType
            # Indexing the project_id as keyword avoids LiteralOutOfBounds when filtering
            store._client.create_payload_index(
                collection_name=store.collection_name,
                field_name="project_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            store._client.create_payload_index(
                collection_name=store.collection_name,
                field_name="file_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload keyword indexes created successfully.")
        except Exception as e:
            logger.warning(f"Failed to create payload indexes: {e}")
            
    logger.info("Collection initialized successfully.")

    # 4. Clear the fallback vectors and metadata
    fallback_path = ROOT_DIR / "data" / "metadata" / "fallback_vectors.jsonl"
    if fallback_path.exists():
        fallback_path.unlink()
        logger.info("Deleted fallback vectors.")

    rag_index_state = ROOT_DIR / "data" / "metadata" / "rag_index_state.json"
    if rag_index_state.exists():
        rag_index_state.unlink()
        logger.info("Cleared RAG index state.")
        
    index_manifest = ROOT_DIR / "data" / "metadata" / "index_manifest.json"
    if index_manifest.exists():
        index_manifest.unlink()
        logger.info("Cleared index manifest.")

    logger.info("Qdrant reset complete! You can now re-run the indexing script.")

if __name__ == "__main__":
    reset_qdrant()
