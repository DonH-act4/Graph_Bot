"""Single-process local PDF worker isolated from the Agent API runtime."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from evidencegraph.extraction import GraphExtractor
from evidencegraph.gemini_extractor import GeminiGraphExtractor
from evidencegraph.ingestion import parse_pdf
from evidencegraph.papers import PaperStore

logger = logging.getLogger(__name__)


class PaperWorker:
    def __init__(self, store: PaperStore, *, graph_extractor: GraphExtractor | None = None) -> None:
        self.store = store
        self.graph_extractor = graph_extractor

    def run_once(self) -> bool:
        """Process at most one queued paper and report whether work was found."""
        for record in self.store.list_processing_records():
            lock_path = self.store.source_path(record.document_id).parent / "worker.lock"
            try:
                lock_path.touch(exist_ok=False)
            except FileExistsError:
                self.store.mark_failed(
                    record.document_id,
                    "PDF worker was interrupted; upload this sample again to retry",
                )
                lock_path.unlink(missing_ok=True)
                return True
            try:
                self.store.process(record.document_id)
            finally:
                lock_path.unlink(missing_ok=True)
            return True
        if self.graph_extractor is None:
            return False
        for record in self.store.list_queued_graph_records():
            lock_path = self.store.source_path(record.document_id).parent / "graph-worker.lock"
            try:
                lock_path.touch(exist_ok=False)
            except FileExistsError:
                self.store.mark_graph_failed(
                    record.document_id,
                    "Graph worker was interrupted; request extraction again to retry",
                )
                lock_path.unlink(missing_ok=True)
                return True
            try:
                self.store.mark_graph_processing(record.document_id)
                self.store.process_graph(record.document_id, self.graph_extractor)
            finally:
                lock_path.unlink(missing_ok=True)
            return True
        return False


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
    root = Path(os.getenv("EVIDENCEGRAPH_DATA_DIR", "data/evidencegraph"))
    graph_model = os.getenv("EVIDENCEGRAPH_GRAPH_MODEL")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    graph_extractor = None
    if graph_model and google_api_key:
        graph_extractor = GeminiGraphExtractor(model=graph_model, api_key=google_api_key)
    worker = PaperWorker(
        PaperStore(root, parser=parse_pdf), graph_extractor=graph_extractor
    )
    while True:
        if not worker.run_once():
            time.sleep(1)


if __name__ == "__main__":
    main()
