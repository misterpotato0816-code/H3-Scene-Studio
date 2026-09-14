// Reusable helpers for driving H3 Phase 1 tests from the ComfyUI frontend.
// Paste into the browser console (or the MCP javascript tool) against the
// running ComfyUI instance, then call h3.runTest(...).
//
// Rationale: graphToPrompt() is a frontend function and is the only thing that
// resolves bypassed nodes correctly, so tests are submitted from the browser
// and polled from Python (build/wait_run.py).

window.h3 = (() => {
  const app = window.comfyAPI.app.app;

  const byTitle = (t) => app.graph._nodes.filter(n => (n.title || '') .includes(t));
  const byType  = (t) => app.graph._nodes.filter(n => n.type === t);

  async function load(name, folder = 'H3') {
    const r = await fetch(`/api/userdata/workflows%2F${folder}%2F${name}.json`);
    if (!r.ok) throw new Error(`load ${name}: HTTP ${r.status}`);
    await app.loadGraphData(JSON.parse(await r.text()), true, false, name);
    await new Promise(s => setTimeout(s, 1500));
    return { nodes: app.graph._nodes.length,
             links: Object.keys(app.graph.links || {}).length };
  }

  // mode: 0 = active, 4 = bypass
  function setMode(titleFragment, mode) {
    const hit = byTitle(titleFragment);
    hit.forEach(n => { n.mode = mode; });
    return hit.map(n => `#${n.id} ${n.type}`);
  }

  function setGroupMode(groupTitleFragment, mode) {
    const g = (app.graph._groups || []).find(g => (g.title || '').includes(groupTitleFragment));
    if (!g) throw new Error(`no group matching ${groupTitleFragment}`);
    g.recomputeInsideNodes?.();
    const inside = g._nodes || g.nodes || [];
    inside.forEach(n => { n.mode = mode; });
    return inside.map(n => `#${n.id} ${n.type}`);
  }

  function setWidget(titleFragment, widgetName, value) {
    const out = [];
    for (const n of byTitle(titleFragment)) {
      const w = (n.widgets || []).find(w => w.name === widgetName);
      if (w) { w.value = value; out.push(`#${n.id} ${n.type}.${widgetName} = ${value}`); }
    }
    if (!out.length) throw new Error(`no widget ${widgetName} on "${titleFragment}"`);
    return out;
  }

  function readWidget(titleFragment, widgetName) {
    for (const n of byTitle(titleFragment)) {
      const w = (n.widgets || []).find(w => w.name === widgetName);
      if (w) return w.value;
    }
    return undefined;
  }

  async function submit(clientId = 'h3-test') {
    const p = await app.graphToPrompt();
    const resp = await fetch('/api/prompt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: p.output, client_id: clientId })
    });
    const body = await resp.json().catch(() => ({}));
    return {
      status: resp.status,
      prompt_id: body.prompt_id,
      apiNodes: Object.keys(p.output).length,
      node_errors: body.node_errors
        ? Object.entries(body.node_errors).map(([id, e]) =>
            ({ id, class_type: e.class_type,
               errors: (e.errors || []).map(x => `${x.type}: ${x.details}`) }))
        : undefined,
      error: body.error
    };
  }

  // Groups are matched by their leading number ("45", "50", ...). LiteGraph's
  // group._nodes is not reliably populated after a programmatic load, so
  // membership is computed from the bounding box instead.
  function groupNodes(prefix) {
    const grp = (app.graph._groups || app.graph.groups || [])
      .find(g => (g.title || '').split(' ')[0] === prefix);
    if (!grp) throw new Error(`no group ${prefix}`);
    const [gx, gy, gw, gh] = [grp.pos[0], grp.pos[1], grp.size[0], grp.size[1]];
    return app.graph._nodes.filter(n =>
      n.pos[0] >= gx && n.pos[1] >= gy && n.pos[0] <= gx + gw && n.pos[1] <= gy + gh);
  }

  function setGroups(prefixes, mode, skipTitleFragments = []) {
    let c = 0;
    for (const p of prefixes)
      for (const n of groupNodes(p)) {
        if (skipTitleFragments.some(f => (n.title || '').includes(f))) continue;
        if (n.mode !== mode) { n.mode = mode; c++; }
      }
    return c;
  }

  // Test 4: continuation. Enables the tail relay and the SYS-B system prompt,
  // and points LoadVideo at the previous clip.
  async function setupContinuation(prevVideoFilename, tailFrames = 56) {
    const on = setGroups(['45'], 0);
    const sysB = byTitle('SYS-B')[0];
    if (sysB) sysB.mode = 0;
    setWidget('前クリップの動画', 'file', prevVideoFilename);
    setWidget('tail フレーム数', 'value', tailFrames);
    return { tailRelayEnabled: on, sysB: !!sysB, prevVideoFilename, tailFrames };
  }

  return { app, byTitle, byType, load, setMode, setGroupMode, groupNodes, setGroups,
           setWidget, readWidget, submit, setupContinuation };
})();
'h3 driver ready';
