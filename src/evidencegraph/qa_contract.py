"""Reviewable question/answer cases tied to exact PDF evidence versions."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from evidencegraph.graph_contract import EvidenceRef
from evidencegraph.models import ParsedDocument

DocumentHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    INSUFFICIENT = "insufficient"


class QAGoldCase(BaseModel):
    """One manually reviewed question within an explicit document scope."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    selected_documents: tuple[DocumentHash, ...] = Field(min_length=1)
    answerability: Answerability
    reference_answer: str | None = None
    supporting_evidence: tuple[EvidenceRef, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer_rule(self) -> QAGoldCase:
        if len(self.selected_documents) != len(set(self.selected_documents)):
            raise ValueError("duplicate selected PDF version")
        if self.answerability is Answerability.ANSWERABLE:
            if not self.reference_answer or not self.reference_answer.strip():
                raise ValueError("answerable case requires a reference answer")
            if not self.supporting_evidence:
                raise ValueError("answerable case requires supporting evidence")
        elif self.reference_answer is not None or self.supporting_evidence:
            raise ValueError("insufficient case cannot claim an answer or supporting evidence")
        return self


class QAGoldSet(BaseModel):
    """Small saved evaluation corpus; not a model response or scoring engine."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    cases: tuple[QAGoldCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> QAGoldSet:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate QA case ID")
        return self

    def validate_against(self, documents: Iterable[ParsedDocument]) -> None:
        """Ensure every answer citation resolves inside its selected PDF versions."""
        document_list = tuple(documents)
        document_index = {document.source_sha256: document for document in document_list}
        if len(document_index) != len(document_list):
            raise ValueError("duplicate PDF version in parsed documents")
        known_blocks = {
            (document.source_sha256, block.block_id)
            for document in document_list
            for block in document.blocks
        }

        for case in self.cases:
            selected = set(case.selected_documents)
            for source_hash in selected:
                if source_hash not in document_index:
                    raise ValueError(f"unknown selected PDF version: {case.case_id}/{source_hash}")
            for ref in case.supporting_evidence:
                if ref.source_sha256 not in selected:
                    raise ValueError(f"evidence outside selected PDFs: {case.case_id}/{ref.block_id}")
                if (ref.source_sha256, ref.block_id) not in known_blocks:
                    raise ValueError(f"unknown evidence block: {case.case_id}/{ref.block_id}")
