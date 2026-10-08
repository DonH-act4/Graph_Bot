export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector('[data-eg-root]');
  const stage = root.querySelector('.eg-graph__stage');
  const svg = root.querySelector('svg');
  const tooltip = root.querySelector('.eg-graph__tooltip');
  const statusLine = root.querySelector('.eg-graph__status');
  const fitButton = root.querySelector('[data-fit]');
  const ns = 'http://www.w3.org/2000/svg';
  const colors = {
    paper: '#8b7cff',
    actor: '#ff8f70',
    concept: '#bd8cff',
    artifact: '#43d4a5',
    process: '#55b8ff',
    observation: '#ffb45c',
    context: '#7d91ae',
    claim: '#ff7195',
    method: '#55b8ff',
    dataset: '#43d4a5',
    result: '#ffb45c',
  };
  const radii = {
    paper: 34, actor: 27, concept: 27, artifact: 28, process: 31,
    observation: 27, claim: 25, context: 25,
    method: 32, dataset: 27, result: 27,
  };
  const source = data?.elements ?? { nodes: [], edges: [] };
  const nodes = source.nodes.map((item) => ({
    ...item.data,
    radius: radii[item.data.type] ?? 33,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    fixed: false,
  }));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges = source.edges.map((item) => ({
    ...item.data,
    sourceNode: byId.get(item.data.source),
    targetNode: byId.get(item.data.target),
  })).filter((edge) => edge.sourceNode && edge.targetNode);
  const degree = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  });

  let width = Math.max(stage.clientWidth, 320);
  let height = stage.clientHeight;
  let transform = { x: 0, y: 0, k: 1 };
  let alpha = 1;
  let frame = 0;
  let dragging = null;
  let dragMoved = false;
  let dragOrigin = null;
  let dragOffset = null;
  let suppressNodeClick = false;
  let panning = null;
  let destroyed = false;
  let ambientFrame = 0;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const create = (tag, attrs = {}) => {
    const element = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  };

  const defs = create('defs');
  const marker = create('marker', {
    id: 'eg-arrow',
    viewBox: '0 0 10 10',
    refX: 9,
    refY: 5,
    markerWidth: 6,
    markerHeight: 6,
    orient: 'auto-start-reverse',
  });
  marker.appendChild(create('path', { d: 'M 0 0 L 10 5 L 0 10 z', class: 'eg-arrow' }));
  defs.appendChild(marker);
  svg.replaceChildren(defs);
  const viewport = create('g', { class: 'eg-viewport' });
  const edgeLayer = create('g');
  const labelLayer = create('g');
  const nodeLayer = create('g');
  viewport.append(edgeLayer, labelLayer, nodeLayer);
  svg.appendChild(viewport);

  const edgeViews = edges.map((edge, edgeIndex) => {
    const line = create('path', {
      class: 'eg-edge',
      'marker-end': 'url(#eg-arrow)',
    });
    const hit = create('path', { class: 'eg-edge__hit' });
    const particle = create('circle', { class: 'eg-edge-particle', r: 2.2 });
    const edgeName = edge.relation_type.replaceAll('_', ' ');
    const label = create('g', {
      class: 'eg-edge-label',
      role: 'button',
      tabindex: 0,
      'aria-label': `${edgeName} relationship`,
    });
    const labelWidth = Math.max(48, edge.relation_type.length * 6.3 + 14);
    const rect = create('rect', { x: -labelWidth / 2, y: -10, width: labelWidth, height: 20, rx: 8 });
    const text = create('text');
    text.textContent = edgeName;
    label.append(rect, text);
    edgeLayer.append(line, particle, hit);
    labelLayer.appendChild(label);
    const select = () => {
      setSelected(edge.id, 'edges');
      setStateValue('selected', { data: { target_id: edge.id, target_group: 'edges' } });
    };
    hit.addEventListener('click', select);
    label.addEventListener('click', select);
    label.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    });
    const show = (event) => showTooltip(event, edgeName, `${edge.evidence_count} source block(s)`);
    hit.addEventListener('pointerenter', (event) => { highlight(edge.id, 'edge'); show(event); });
    label.addEventListener('pointerenter', (event) => { highlight(edge.id, 'edge'); show(event); });
    hit.addEventListener('pointermove', moveTooltip);
    label.addEventListener('pointermove', moveTooltip);
    hit.addEventListener('pointerleave', clearHover);
    label.addEventListener('pointerleave', clearHover);
    return { edge, edgeIndex, line, hit, particle, label };
  });

  const wrapLabel = (value) => {
    const normalized = value.replace(/\s+/g, ' ').trim();
    if (normalized.length <= 14) return [normalized];
    const words = normalized.split(' ');
    const lines = ['', ''];
    for (const word of words) {
      const index = lines[0].length < 12 ? 0 : 1;
      const candidate = `${lines[index]} ${word}`.trim();
      if (index === 1 && candidate.length > 17) {
        lines[1] = `${candidate.slice(0, 15)}…`;
        break;
      }
      lines[index] = candidate;
    }
    return lines.filter(Boolean).slice(0, 2);
  };

  const nodeViews = nodes.map((node) => {
    const typeName = node.domain_type ?? node.type;
    const group = create('g', {
      class: 'eg-node',
      role: 'button',
      tabindex: 0,
      'aria-label': `${node.name}, ${typeName}`,
    });
    const circle = create('circle', { r: node.radius, fill: colors[node.type] ?? '#64748b' });
    const title = create('title');
    title.textContent = `${node.name} · ${typeName} · ${node.evidence_count} evidence block(s)`;
    const text = create('text');
    wrapLabel(node.display_name ?? node.name).forEach((line, index, lines) => {
      const span = create('tspan', { x: 0, dy: index === 0 ? (lines.length === 1 ? 0 : -6) : 13 });
      span.textContent = line;
      text.appendChild(span);
    });
    group.append(circle, title, text);
    nodeLayer.appendChild(group);

    group.addEventListener('pointerenter', (event) => {
      highlight(node.id, 'node');
      showTooltip(event, node.name, `${typeName} · ${node.evidence_count} evidence block(s)`);
    });
    group.addEventListener('pointermove', moveTooltip);
    group.addEventListener('pointerleave', clearHover);
    group.addEventListener('pointerdown', (event) => {
      event.stopPropagation();
      group.setPointerCapture(event.pointerId);
      const point = graphPoint(event);
      dragging = node;
      dragMoved = false;
      dragOrigin = point;
      dragOffset = { x: node.x - point.x, y: node.y - point.y };
      suppressNodeClick = false;
      node.fixed = true;
      group.classList.add('is-dragging');
      alpha = Math.max(alpha, .42);
    });
    group.addEventListener('pointermove', (event) => {
      if (dragging !== node) return;
      const point = graphPoint(event);
      dragMoved = dragMoved || Math.hypot(dragOrigin.x - point.x, dragOrigin.y - point.y) > 3;
      if (!dragMoved) return;
      node.x = point.x + dragOffset.x;
      node.y = point.y + dragOffset.y;
      node.vx = 0;
      node.vy = 0;
      alpha = Math.max(alpha, .28);
      start();
    });
    group.addEventListener('click', () => {
      if (suppressNodeClick) {
        suppressNodeClick = false;
        return;
      }
      setSelected(node.id, 'nodes');
      setStateValue('selected', { data: { target_id: node.id, target_group: 'nodes' } });
    });
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        setSelected(node.id, 'nodes');
        setStateValue('selected', { data: { target_id: node.id, target_group: 'nodes' } });
      }
    });
    const finishDrag = () => {
      if (dragging !== node) return;
      suppressNodeClick = dragMoved;
      node.fixed = false;
      dragging = null;
      dragOrigin = null;
      dragOffset = null;
      group.classList.remove('is-dragging');
      alpha = Math.max(alpha, .36);
      start();
    };
    group.addEventListener('pointerup', finishDrag);
    group.addEventListener('pointercancel', finishDrag);
    return { node, group };
  });

  function seed() {
    width = Math.max(stage.clientWidth, 320);
    height = stage.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const ordered = [...nodes].sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0));
    const hub = ordered.shift();
    if (hub) {
      hub.x = width / 2;
      hub.y = height / 2;
    }
    const baseRadius = Math.min(width * .24, height * .27);
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    ordered.forEach((node, index) => {
      const angle = -Math.PI / 2 + index * goldenAngle;
      const ring = baseRadius * (1 + (index % 3) * .22);
      node.x = width / 2 + Math.cos(angle) * ring;
      node.y = height / 2 + Math.sin(angle) * ring * .82;
    });
    alpha = 1;
    start();
  }

  function simulate() {
    if (destroyed) return;
    const centerX = width / 2;
    const centerY = height / 2;
    nodes.forEach((node) => {
      if (node.fixed) return;
      node.vx += (centerX - node.x) * .0007 * alpha;
      node.vy += (centerY - node.y) * .0007 * alpha;
    });
    edges.forEach((edge) => {
      const a = edge.sourceNode;
      const b = edge.targetNode;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const desired = 132 + (a.radius + b.radius) * .42;
      const force = (distance - desired) * .0042 * alpha;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      if (!a.fixed) { a.vx += fx; a.vy += fy; }
      if (!b.fixed) { b.vx -= fx; b.vy -= fy; }
    });
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distance = Math.max(Math.hypot(dx, dy), .1);
        const minimum = a.radius + b.radius + 22;
        let repel = Math.min(2.2, 1550 / (distance * distance)) * alpha;
        if (distance < minimum) repel += (minimum - distance) * .018 * alpha;
        const fx = (dx / distance) * repel;
        const fy = (dy / distance) * repel;
        if (!a.fixed) { a.vx -= fx; a.vy -= fy; }
        if (!b.fixed) { b.vx += fx; b.vy += fy; }
      }
    }
    nodes.forEach((node) => {
      if (!node.fixed) {
        node.vx *= .84;
        node.vy *= .84;
        node.x += node.vx;
        node.y += node.vy;
      }
      const pad = node.radius + 18;
      node.x = Math.max(pad, Math.min(width - pad, node.x));
      node.y = Math.max(pad, Math.min(height - pad, node.y));
    });
    alpha *= .972;
    render();
    if (alpha > .006 || dragging) frame = requestAnimationFrame(simulate);
    else frame = 0;
  }

  function render(timestamp = performance.now()) {
    viewport.setAttribute('transform', `translate(${transform.x} ${transform.y}) scale(${transform.k})`);
    edgeViews.forEach(({ edge, edgeIndex, line, hit, particle, label }) => {
      const { sourceNode: a, targetNode: b } = edge;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const ux = dx / distance;
      const uy = dy / distance;
      const x1 = a.x + ux * (a.radius + 3);
      const y1 = a.y + uy * (a.radius + 3);
      const x2 = b.x - ux * (b.radius + 10);
      const y2 = b.y - uy * (b.radius + 10);
      const bend = ((edgeIndex % 2 === 0 ? 1 : -1) * (9 + (edgeIndex % 3) * 4));
      const cx = (x1 + x2) / 2 - uy * bend;
      const cy = (y1 + y2) / 2 + ux * bend;
      const pathData = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
      line.setAttribute('d', pathData);
      hit.setAttribute('d', pathData);
      label.setAttribute('transform', `translate(${(x1 + 2 * cx + x2) / 4} ${(y1 + 2 * cy + y2) / 4})`);
      const progress = reduceMotion ? .64 : ((timestamp / 3200) + edgeIndex / Math.max(edges.length, 1)) % 1;
      const inverse = 1 - progress;
      const px = inverse * inverse * x1 + 2 * inverse * progress * cx + progress * progress * x2;
      const py = inverse * inverse * y1 + 2 * inverse * progress * cy + progress * progress * y2;
      particle.setAttribute('cx', px);
      particle.setAttribute('cy', py);
    });
    nodeViews.forEach(({ node, group }) => group.setAttribute('transform', `translate(${node.x} ${node.y})`));
  }

  function start() {
    if (!frame) frame = requestAnimationFrame(simulate);
  }

  function animate(timestamp) {
    if (destroyed) return;
    if (!frame) render(timestamp);
    ambientFrame = requestAnimationFrame(animate);
  }

  function graphPoint(event) {
    const rect = svg.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
      x: ((event.clientX - rect.left) * scaleX - transform.x) / transform.k,
      y: ((event.clientY - rect.top) * scaleY - transform.y) / transform.k,
    };
  }

  function setSelected(id, group) {
    nodeViews.forEach(({ node, group: element }) => element.classList.toggle('is-active', group === 'nodes' && node.id === id));
    edgeViews.forEach(({ edge, line }) => line.classList.toggle('is-active', group === 'edges' && edge.id === id));
    statusLine.textContent = group === 'nodes' ? 'Entity selected · source text below.' : 'Relationship selected · source text below.';
  }

  function highlight(id, kind) {
    const connectedNodes = new Set();
    const connectedEdges = new Set();
    if (kind === 'node') {
      connectedNodes.add(id);
      edges.forEach((edge) => {
        if (edge.source === id || edge.target === id) {
          connectedEdges.add(edge.id);
          connectedNodes.add(edge.source);
          connectedNodes.add(edge.target);
        }
      });
    } else {
      connectedEdges.add(id);
      const edge = edges.find((item) => item.id === id);
      if (edge) { connectedNodes.add(edge.source); connectedNodes.add(edge.target); }
    }
    nodeViews.forEach(({ node, group }) => group.classList.toggle('is-dim', !connectedNodes.has(node.id)));
    edgeViews.forEach(({ edge, line, label }) => {
      const dim = !connectedEdges.has(edge.id);
      line.classList.toggle('is-dim', dim);
      label.classList.toggle('is-dim', dim);
    });
  }

  function clearHover() {
    tooltip.hidden = true;
    nodeViews.forEach(({ group }) => group.classList.remove('is-dim'));
    edgeViews.forEach(({ line, label }) => {
      line.classList.remove('is-dim');
      label.classList.remove('is-dim');
    });
  }

  function showTooltip(event, heading, detail) {
    tooltip.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = heading;
    const line = document.createElement('div');
    line.textContent = detail;
    tooltip.append(strong, line);
    tooltip.hidden = false;
    moveTooltip(event);
  }

  function moveTooltip(event) {
    const rect = stage.getBoundingClientRect();
    tooltip.style.left = `${Math.min(event.clientX - rect.left + 14, rect.width - 274)}px`;
    tooltip.style.top = `${Math.max(10, event.clientY - rect.top + 14)}px`;
  }

  function fit() {
    transform = { x: 0, y: 0, k: 1 };
    render();
  }

  function onWheel(event) {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const px = (event.clientX - rect.left) * width / rect.width;
    const py = (event.clientY - rect.top) * height / rect.height;
    const next = Math.max(.65, Math.min(2.2, transform.k * Math.exp(-event.deltaY * .001)));
    transform.x = px - ((px - transform.x) / transform.k) * next;
    transform.y = py - ((py - transform.y) / transform.k) * next;
    transform.k = next;
    render();
  }

  function onBackgroundDown(event) {
    if (event.target !== svg) return;
    panning = { x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y };
    svg.setPointerCapture(event.pointerId);
  }

  function onBackgroundMove(event) {
    if (!panning) return;
    transform.x = panning.tx + event.clientX - panning.x;
    transform.y = panning.ty + event.clientY - panning.y;
    render();
  }

  function onBackgroundUp() { panning = null; }

  svg.addEventListener('wheel', onWheel, { passive: false });
  svg.addEventListener('pointerdown', onBackgroundDown);
  svg.addEventListener('pointermove', onBackgroundMove);
  svg.addEventListener('pointerup', onBackgroundUp);
  svg.addEventListener('pointercancel', onBackgroundUp);
  fitButton.addEventListener('click', fit);
  const observer = new ResizeObserver(() => { seed(); });
  observer.observe(stage);
  seed();
  if (!reduceMotion) ambientFrame = requestAnimationFrame(animate);
  if (data?.selected_id) {
    setSelected(data.selected_id, data.selected_id.startsWith('edge:') ? 'edges' : 'nodes');
  }

  return () => {
    destroyed = true;
    cancelAnimationFrame(frame);
    cancelAnimationFrame(ambientFrame);
    observer.disconnect();
    svg.removeEventListener('wheel', onWheel);
    svg.removeEventListener('pointerdown', onBackgroundDown);
    svg.removeEventListener('pointermove', onBackgroundMove);
    svg.removeEventListener('pointerup', onBackgroundUp);
    svg.removeEventListener('pointercancel', onBackgroundUp);
    fitButton.removeEventListener('click', fit);
  };
}
