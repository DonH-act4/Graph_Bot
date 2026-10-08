"""Automatic paper context retrieves real, bounded passages for common questions."""

from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.retrieval import is_contextual_followup, retrieve_paper_blocks


def paper() -> ParsedDocument:
    entries = (
        ("title", "A catalyst for water purification and pollutant removal"),
        ("section_header", "Abstract"),
        ("text", "We developed a reusable catalyst to remove organic pollutants from water."),
        ("section_header", "2. Materials and methods"),
        ("text", "The catalyst was prepared by heating a precursor at 400 degrees for two hours."),
        ("text", "Scanning microscopy and spectroscopy were used to characterize the material."),
        ("section_header", "3. Results and discussion"),
        ("text", "Pollutant removal reached 95 percent after 30 minutes under visible light."),
        ("text", "The catalyst retained stable performance across five recycling experiments."),
        ("section_header", "4. Conclusion"),
        ("text", "The reusable material provides an efficient approach to water purification."),
        ("section_header", "References"),
        ("text", "Previous catalyst studies report 100 percent removal in a different system."),
    )
    blocks = tuple(
        ParsedBlock(
            block_id=f"blk_{index:024x}",
            source_ref=f"#/texts/{index}",
            label=label,
            text=text,
            locations=(SourceLocation(
                page_number=index // 4 + 1,
                bounding_box=BoundingBox(
                    left=0, top=0, right=1, bottom=1, coordinate_origin="bottom-left"
                ),
                character_start=0,
                character_end=len(text),
            ),),
        )
        for index, (label, text) in enumerate(entries)
    )
    return ParsedDocument(
        source_filename="paper.pdf", source_sha256="a" * 64,
        parser_version="test", page_count=4, blocks=blocks,
    )


def test_chinese_overview_prioritizes_abstract_and_conclusion() -> None:
    document = paper()
    selected = retrieve_paper_blocks(document, "这篇论文做了什么？", max_blocks=3)
    assert {block.block_id for block in selected} == {
        document.blocks[index].block_id for index in (0, 2, 10)
    }


def test_chinese_methods_and_results_retrieve_different_sections() -> None:
    document = paper()
    methods = retrieve_paper_blocks(document, "采用了什么方法？", max_blocks=2)
    results = retrieve_paper_blocks(document, "主要结果和发现是什么？", max_blocks=2)
    assert document.blocks[4] in methods
    assert document.blocks[7] in results
    assert document.blocks[4] not in results


def test_specific_english_terms_rank_source_passage_and_exclude_references() -> None:
    document = paper()
    selected = retrieve_paper_blocks(document, "What did scanning spectroscopy characterize?", max_blocks=1)
    assert selected == (document.blocks[5],)
    assert document.blocks[-1] not in retrieve_paper_blocks(document, "catalyst", max_blocks=20)


def test_retrieval_enforces_character_budget_without_truncating_source_text() -> None:
    document = paper()
    selected = retrieve_paper_blocks(document, "主要结果", max_blocks=10, max_characters=160)
    assert sum(len(block.text) for block in selected) <= 160
    assert selected
    assert all(block in document.blocks for block in selected)


def test_short_contextual_followups_are_distinct_from_new_questions() -> None:
    assert is_contextual_followup("为什么？")
    assert is_contextual_followup("解释一下这个")
    assert not is_contextual_followup("论文采用了什么方法？")


def test_chinese_transfer_metrics_and_numeric_query_retrieve_later_results() -> None:
    document = paper()
    template = document.blocks[7]

    def result_block(index: int, text: str) -> ParsedBlock:
        return template.model_copy(update={
            "block_id": f"blk_{index:024x}",
            "source_ref": f"#/texts/{index}",
            "text": text,
            "locations": (template.locations[0].model_copy(update={
                "character_end": len(text),
            }),),
        })

    fillers = tuple(result_block(
        100 + index,
        f"The experiment examined the general network behavior in condition {index + 1}.",
    ) for index in range(15))
    target = result_block(
        200,
        "Transfer learning was evaluated across 40 scenarios. Classification accuracy "
        "improved from 82.5 to 97.5 percent, and mean absolute error decreased "
        "from 3.2 to 0.8 meters.",
    )
    number_only_target = result_block(
        201,
        "A separate evaluation was conducted using 40 scenarios to assess model robustness.",
    )
    document = document.model_copy(update={
        "blocks": (document.blocks[0], document.blocks[6], *fillers, target, number_only_target),
    })
    selected = retrieve_paper_blocks(
        document,
        "迁移学习在40个场景测试中带来了什么提升？请给出分类准确率和漏点定位误差的前后数值。",
    )
    assert target in selected
    assert number_only_target in selected
    assert len(selected) <= 10


def test_spaced_abstract_heading_is_recognized_for_overview() -> None:
    document = paper()
    spaced_header = document.blocks[1].model_copy(update={"text": "A B S T R A C T"})
    document = document.model_copy(update={
        "blocks": (document.blocks[0], spaced_header, *document.blocks[2:]),
    })
    selected = retrieve_paper_blocks(document, "论文做了什么？", max_blocks=3)
    assert document.blocks[2] in selected
