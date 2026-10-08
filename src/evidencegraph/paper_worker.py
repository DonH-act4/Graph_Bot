"""Single-process local PDF worker isolated from the Agent API runtime."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from evidencegraph.extraction import GraphExtractionError, GraphExtractor
from evidencegraph.gemini_extractor import GeminiGraphExtractor
from evidencegraph.graph_models import graph_model_provider, provider_model_name
from evidencegraph.groq_extractor import GroqGraphExtractor
from evidencegraph.ingestion import parse_pdf
from evidencegraph.ollama_extractor import OllamaGraphExtractor
from evidencegraph.papers import PaperDeletionRequested, PaperStore

logger = logging.getLogger(__name__)


class PaperWorker:
    def __init__(
        self,
        store: PaperStore,
        *,
        graph_extractor: GraphExtractor | None = None,
        graph_extractor_factory: Callable[[str], GraphExtractor] | None = None,
        isolate_jobs: bool = False,
    ) -> None:
        self.store = store
        self.graph_extractor = graph_extractor
        self.graph_extractor_factory = graph_extractor_factory
        self.isolate_jobs = isolate_jobs

    def _run_isolated(self, document_id: str, kind: str) -> None:
        """Keep the worker responsive to deletion while Docling/model calls run."""
        process = subprocess.Popen(
            [sys.executable, "-m", "evidencegraph.paper_worker", "--job", kind, document_id],
            start_new_session=(os.name == "posix"),
        )
        try:
            while process.poll() is None:
                if self.store.deletion_state(document_id) is not None:
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait()
                    return
                time.sleep(0.1)
            if process.returncode and self.store.deletion_state(document_id) is None:
                if kind == "pdf":
                    self.store.mark_failed(document_id, "PDF worker exited unexpectedly")
                else:
                    self.store.mark_graph_failed(document_id, "Graph worker exited unexpectedly")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()

    def run_once(self) -> bool:
        """Process at most one queued paper and report whether work was found."""
        # If a previous worker died after deletion was requested, its locks are stale.
        for document_id in self.store.list_deleting_documents():
            folder = self.store.source_path(document_id).parent
            for lock_name in ("worker.lock", "graph-worker.lock"):
                (folder / lock_name).unlink(missing_ok=True)
        for record in self.store.list_processing_records():
            lock_path = self.store.source_path(record.document_id).parent / "worker.lock"
            try:
                if not self.store.claim_job(record.document_id, "worker.lock"):
                    continue
            except FileExistsError:
                if self.store.deletion_state(record.document_id) is None:
                    self.store.mark_failed(
                        record.document_id,
                        "PDF worker was interrupted; upload this sample again to retry",
                    )
                lock_path.unlink(missing_ok=True)
                return True
            try:
                if self.isolate_jobs:
                    self._run_isolated(record.document_id, "pdf")
                elif self.store.deletion_state(record.document_id) is None:
                    self.store.process(record.document_id)
            finally:
                lock_path.unlink(missing_ok=True)
            return True
        if self.graph_extractor is None and self.graph_extractor_factory is None:
            return False
        for record in self.store.list_queued_graph_records():
            lock_path = self.store.source_path(record.document_id).parent / "graph-worker.lock"
            try:
                if not self.store.claim_job(record.document_id, "graph-worker.lock"):
                    continue
            except FileExistsError:
                if self.store.deletion_state(record.document_id) is None:
                    self.store.mark_graph_failed(
                        record.document_id,
                        "Graph worker was interrupted; request extraction again to retry",
                    )
                lock_path.unlink(missing_ok=True)
                return True
            try:
                if self.isolate_jobs:
                    self._run_isolated(record.document_id, "graph")
                    return True
                if self.store.deletion_state(record.document_id) is not None:
                    return True
                extractor = self.graph_extractor
                if self.graph_extractor_factory is not None:
                    if record.requested_model is None:
                        self.store.mark_graph_failed(
                            record.document_id,
                            "Graph task does not specify an extraction model",
                        )
                        return True
                    try:
                        extractor = self.graph_extractor_factory(record.requested_model)
                    except GraphExtractionError as exc:
                        self.store.mark_graph_failed(record.document_id, str(exc))
                        return True
                if extractor is None:
                    return False
                try:
                    self.store.mark_graph_processing(record.document_id)
                    self.store.process_graph(record.document_id, extractor)
                except PaperDeletionRequested:
                    pass
            finally:
                lock_path.unlink(missing_ok=True)
            return True
        return False


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
    root = Path(os.getenv("EVIDENCEGRAPH_DATA_DIR", "data/evidencegraph"))
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    extractor_factory: Callable[[str], GraphExtractor] | None = None
    if google_api_key or groq_api_key or ollama_base_url:
        def create_graph_extractor(model: str) -> GraphExtractor:
            provider = graph_model_provider(model)
            provider_model = provider_model_name(model)
            if provider == "groq":
                if not groq_api_key:
                    raise GraphExtractionError(
                        "Groq graph extraction requires GROQ_API_KEY"
                    )
                return GroqGraphExtractor(model=provider_model, api_key=groq_api_key)
            if provider == "ollama":
                if not ollama_base_url:
                    raise GraphExtractionError(
                        "Ollama graph extraction requires OLLAMA_BASE_URL"
                    )
                return OllamaGraphExtractor(model=provider_model, base_url=ollama_base_url)
            if not google_api_key:
                raise GraphExtractionError(
                    "Gemini graph extraction requires GOOGLE_API_KEY"
                )
            return GeminiGraphExtractor(model=provider_model, api_key=google_api_key)

        extractor_factory = create_graph_extractor
    worker = PaperWorker(
        PaperStore(root, parser=parse_pdf),
        graph_extractor_factory=extractor_factory,
        isolate_jobs=len(sys.argv) == 1,
    )
    if len(sys.argv) == 4 and sys.argv[1] == "--job":
        _, _, kind, document_id = sys.argv
        if kind == "pdf":
            worker.store.process(document_id)
        elif kind == "graph":
            record = worker.store.get_graph_record(document_id)
            if record is None or record.requested_model is None or extractor_factory is None:
                worker.store.mark_graph_failed(document_id, "Graph model is unavailable")
                return
            extractor = extractor_factory(record.requested_model)
            worker.store.mark_graph_processing(document_id)
            worker.store.process_graph(document_id, extractor)
        else:
            raise ValueError("Unknown worker job")
        return
    while True:
        if not worker.run_once():
            time.sleep(1)


if __name__ == "__main__":
    main()
