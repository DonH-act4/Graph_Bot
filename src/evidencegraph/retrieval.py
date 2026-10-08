"""Small, deterministic retrieval baseline for conversations about one paper."""

from __future__ import annotations

import math
import re
from collections import Counter

from evidencegraph.models import ParsedBlock, ParsedDocument

MAX_EVIDENCE_BLOCKS = 10
MAX_EVIDENCE_CHARACTERS = 18_000

_WORDS = re.compile(r"[a-z][a-z0-9-]{2,}|\d+(?:\.\d+)?")
_STOP_WORDS = frozenset(
    "the and for with this that from what which how does did are was were can "
    "you paper study research please explain main about into their these its".split()
)
_SECTION_TERMS = {
    "abstract": ("abstract", "summary", "摘要"),
    "introduction": ("introduction", "background", "引言", "背景"),
    "methods": (
        "method", "material", "experimental", "experiment", "procedure", "design",
        "方法", "材料", "实验", "设计",
    ),
    "results": ("result", "discussion", "finding", "结果", "讨论", "发现"),
    "conclusion": ("conclusion", "concluding", "结论", "总结"),
}
_QUERY_TERMS = {
    "overview": (
        "做了什么", "讲什么", "讲了什么", "主要工作", "贡献", "介绍", "总结",
        "这篇", "overview", "summary", "contribution", "about",
    ),
    "methods": ("方法", "怎么做", "如何做", "实验", "材料", "设计", "流程", "method", "approach"),
    "results": ("结果", "发现", "结论", "效果", "性能", "提升", "result", "finding", "conclusion"),
}
_QUERY_EXPANSIONS = {
    "方法": ("method", "approach", "procedure", "experiment"),
    "材料": ("material", "sample", "preparation"),
    "结果": ("result", "performance", "finding"),
    "发现": ("result", "finding", "observed"),
    "结论": ("conclusion", "demonstrate", "suggest"),
    "机制": ("mechanism", "pathway", "reaction"),
    "污染": ("pollutant", "contaminant", "removal"),
    "催化": ("catalyst", "catalytic", "catalysis"),
    "效率": ("efficiency", "performance", "rate"),
    "迁移": ("transfer", "adaptation", "fine-tuning"),
    "准确率": ("accuracy", "classification"),
    "误差": ("error", "mae", "mse"),
    "提升": ("improvement", "improved"),
    "限制": ("limitation", "challenge", "future"),
}


def _terms(text: str) -> set[str]:
    return {word for word in _WORDS.findall(text.casefold()) if word not in _STOP_WORDS}


def _section(text: str) -> str | None:
    lowered = text.casefold()
    compact = re.sub(r"\s+", "", lowered)
    for name, terms in _SECTION_TERMS.items():
        if any(term in lowered or term in compact for term in terms):
            return name
    return None


def is_contextual_followup(query: str) -> bool:
    """Recognize short follow-ups which usually refer to the selected passage."""
    normalized = re.sub(r"[\s，。！？,.!?]", "", query.casefold())
    return len(normalized) <= 28 and any(
        term in normalized
        for term in (
            "为什么", "怎么理解", "解释一下", "详细说", "展开", "这个", "这段",
            "继续", "why", "explainthis", "tellmemore", "whataboutthis",
        )
    )


def retrieve_paper_blocks(
    document: ParsedDocument,
    query: str,
    *,
    max_blocks: int = MAX_EVIDENCE_BLOCKS,
    max_characters: int = MAX_EVIDENCE_CHARACTERS,
) -> tuple[ParsedBlock, ...]:
    """Rank real source blocks by section intent and lexical overlap, within a budget.

    Whole blocks are returned. Reference sections and page furniture are excluded;
    long blocks which do not fit are skipped rather than silently changing source text.
    """
    if max_blocks < 1 or max_characters < 1:
        raise ValueError("Retrieval budgets must be positive")

    normalized_query = query.casefold()
    intent = {
        name for name, terms in _QUERY_TERMS.items()
        if any(term in normalized_query for term in terms)
    }
    if not intent:
        intent = {"overview"}
    query_terms = _terms(query)
    for chinese, expansion in _QUERY_EXPANSIONS.items():
        if chinese in query:
            query_terms.update(expansion)

    candidates: list[tuple[int, ParsedBlock, str | None, set[str]]] = []
    current_section: str | None = None
    for index, block in enumerate(document.blocks):
        label = block.label.casefold()
        text = block.text.strip()
        if label in {"page_header", "page_footer"}:
            continue
        if label in {"section_header", "title"}:
            heading = re.sub(r"^[\d.\s]+", "", text.casefold()).strip()
            if heading in {"references", "bibliography", "参考文献"}:
                break
            current_section = _section(text) or current_section
        if len(text) < 35 and label not in {"title", "table"}:
            continue
        candidates.append((index, block, current_section, _terms(text)))

    frequency = Counter(term for _, _, _, terms in candidates for term in terms)
    ranked: list[tuple[float, int, ParsedBlock]] = []
    for index, block, section, terms in candidates:
        score = sum(
            4.0 * math.log(1 + len(candidates) / frequency[term])
            for term in query_terms & terms
        )
        if block.label.casefold() == "title":
            score += 14.0
        if "overview" in intent:
            score += {"abstract": 12.0, "conclusion": 11.0, "introduction": 3.0}.get(section or "", 0.0)
        if "methods" in intent:
            score += {"methods": 16.0, "abstract": 4.0}.get(section or "", 0.0)
        if "results" in intent:
            score += {"results": 16.0, "conclusion": 12.0, "abstract": 4.0}.get(section or "", 0.0)
        if block.label.casefold() == "section_header":
            score -= 8.0
        # A no-heading PDF still gets a bounded opening/concluding baseline.
        if section is None and block.label.casefold() != "title":
            score += max(0.0, 3.0 - index / 20)
        ranked.append((score, index, block))

    selected: list[tuple[int, ParsedBlock]] = []
    used_characters = 0
    for _, index, block in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_blocks:
            break
        if used_characters + len(block.text) > max_characters:
            continue
        selected.append((index, block))
        used_characters += len(block.text)
    return tuple(block for _, block in sorted(selected))
