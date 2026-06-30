from __future__ import annotations

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import settings
from backend.rag.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info(f"Target Qdrant URL: {settings.qdrant_url}")
    
    store = QdrantStore()
    
    if store.wait_for_qdrant(retries=3, base_delay=1.0):
        logger.info("Qdrant is reachable and ready!")
        try:
            info = store.collection_info()
            logger.info(f"Collection: {store.collection_name}")
            logger.info(f"Status: {info.get('status', 'OK')}")
            result = info.get("result", {})
            if result:
                logger.info(f"Vectors count: {result.get('points_count', 0)}")
                logger.info(f"Vector size: {store.vector_size()}")
        except Exception as e:
            logger.warning(f"Could not retrieve collection info (may not exist yet): {e}")
    else:
        logger.error("Failed to reach Qdrant. Please ensure the Docker container is running:")
        logger.error("  docker compose up -d qdrant")
        sys.exit(1)

if __name__ == "__main__":
    main()
