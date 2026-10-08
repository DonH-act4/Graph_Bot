"""Project-owned interactive evidence graph component."""

from pathlib import Path
from typing import Any

import streamlit as st

_ASSET_DIR = Path(__file__).parent
_GRAPH_HTML = """
<section class="eg-graph" data-eg-root aria-label="Interactive evidence graph">
  <div class="eg-graph__stage">
    <svg role="img" aria-label="Evidence graph with draggable nodes and relationships"></svg>
    <div class="eg-graph__tooltip" role="tooltip" hidden></div>
    <div class="eg-graph__controls" aria-label="Graph view controls">
      <button type="button" data-fit aria-label="Fit graph to view">Fit</button>
    </div>
  </div>
  <p class="eg-graph__status" aria-live="polite">
    Drag to reshape · select a relationship to reveal its source.
  </p>
</section>
"""

def render_force_graph(
    elements: dict[str, list[dict[str, dict[str, Any]]]],
    *,
    selected_id: str | None,
    key: str,
) -> object:
    """Render the force graph and return its persistent selection state."""
    force_graph = st.components.v2.component(
        "evidence_force_graph",
        html=_GRAPH_HTML,
        css=(_ASSET_DIR / "graph_component.css").read_text(),
        js=(_ASSET_DIR / "graph_component.js").read_text(),
    )
    return force_graph(
        data={"elements": elements, "selected_id": selected_id},
        key=key,
        width="stretch",
        height=620,
        on_selected_change=lambda: None,
    )
