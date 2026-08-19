(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    progress: $("progress"),
    clock: $("clock"),
    dot: $("dot"),
    stageMeta: $("stage-meta"),
    abstract: $("abstract"),
    scientist: $("scientist"),
    scientistText: $("scientist-text"),
    heroOps: $("hero-ops"),
    edges: $("edges"),
    nodes: $("nodes"),
    graph: $("graph"),
    phys: $("phys-keys"),
    nowName: $("now-name"),
    nowTech: $("now-tech"),
    traceLog: $("trace-log"),
    scoreVal: $("score-val"),
    scoreLabel: $("score-label"),
    opsList: $("ops-list"),
    compare: $("compare"),
  };

  const NODE_W = 220;
  const NODE_H = 96;
  const GAP_X = 40;
  const GAP_Y = 48;

  let run = null;
  let playing = true;
  let t = 0;
  let last = 0;
  let applied = -1;
  let instant = false;

  const live = new Map();
  let gen = 0;

  function hideHeroOps() {
    els.heroOps.hidden = true;
    els.heroOps.innerHTML = "";
  }

  function showHeroOps(ops) {
    if (!ops || !ops.length) {
      hideHeroOps();
      return;
    }
    els.heroOps.innerHTML = ops
      .map((o) => '<span class="' + o.op + '"><span class="m">' + o.mark + "</span> " + o.key + "</span>")
      .join("");
    els.heroOps.hidden = false;
  }

  function resetVisual() {
    gen += 1;
    live.clear();
    els.graph.classList.remove("reading");
    els.nodes.innerHTML = "";
    els.edges.innerHTML = "";
    els.phys.innerHTML = "";
    els.traceLog.textContent = "";
    els.abstract.classList.remove("show");
    els.abstract.innerHTML = "";
    els.scientist.hidden = true;
    hideHeroOps();
    els.compare.classList.remove("hot");
    els.scoreVal.textContent = "\u2013";
    els.scoreVal.className = "score empty";
    els.scoreLabel.textContent = "";
    els.opsList.innerHTML = "";
    els.nowName.textContent = "cold open";
    els.nowTech.textContent = "declare a society";
    els.stageMeta.textContent = "desired vs current";
    document.body.classList.add("cold");
    applied = -1;
  }

  function layout(nodes) {
    const byKey = new Map(nodes.map((n) => [n.key, n]));
    const kids = new Map();
    const roots = [];
    for (const n of nodes) {
      if (n.parentKey && byKey.has(n.parentKey)) {
        if (!kids.has(n.parentKey)) kids.set(n.parentKey, []);
        kids.get(n.parentKey).push(n);
      } else {
        roots.push(n);
      }
    }
    const widthOf = (n) => {
      const ch = kids.get(n.key) ?? [];
      if (!ch.length) return NODE_W;
      return Math.max(
        NODE_W,
        ch.reduce((s, c) => s + widthOf(c), 0) + GAP_X * (ch.length - 1),
      );
    };
    const placed = [];
    const place = (n, x, y) => {
      const w = widthOf(n);
      placed.push({
        ...n,
        x: x + w / 2 - NODE_W / 2,
        y,
      });
      let cx = x;
      for (const c of kids.get(n.key) ?? []) {
        const cw = widthOf(c);
        place(c, cx, y + NODE_H + GAP_Y);
        cx += cw + GAP_X;
      }
    };
    let x = 0;
    for (const r of roots) {
      const w = widthOf(r);
      place(r, x, 0);
      x += w + GAP_X;
    }
    return placed;
  }

  function center(placed) {
    const box = els.graph.getBoundingClientRect();
    if (!placed.length || !box.width) return placed;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of placed) {
      minX = Math.min(minX, n.x);
      maxX = Math.max(maxX, n.x + NODE_W);
      minY = Math.min(minY, n.y);
      maxY = Math.max(maxY, n.y + NODE_H);
    }
    const reserveTop = 36;
    const reserveBot = els.heroOps.hidden ? 28 : 56;
    const ox = (box.width - (maxX - minX)) / 2 - minX;
    const oy = Math.max(reserveTop, (box.height - reserveBot - (maxY - minY)) / 2 - minY);
    return placed.map((n) => ({ ...n, x: n.x + ox, y: n.y + oy }));
  }

  function renderEdges() {
    const items = [...live.values()].filter((n) => n.el && !n.gone);
    const lines = [];
    for (const n of items) {
      if (!n.parentKey) continue;
      const p = live.get(n.parentKey);
      if (!p || p.gone) continue;
      const x1 = p.x + NODE_W / 2;
      const y1 = p.y + NODE_H;
      const x2 = n.x + NODE_W / 2;
      const y2 = n.y;
      const liveEdge = n.pulse || p.pulse;
      lines.push(
        '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + (liveEdge ? "live" : "") + '"/>',
      );
    }
    els.edges.innerHTML = lines.join("");
  }

  function renderPhys() {
    const keys = [...live.values()].filter((n) => !n.gone);
    els.phys.innerHTML = keys
      .map((n) => '<span class="' + (n.badge === "mount" ? "in" : "") + '">' + n.key + "</span>")
      .join("");
  }

  function ensureNode(snap, badge) {
    let n = live.get(snap.key);
    if (!n) {
      const el = document.createElement("div");
      el.className = "node";
      el.dataset.key = snap.key;
      el.dataset.role = snap.role;
      el.innerHTML = '<span class="role">' + snap.role + '</span><span class="key">' + snap.key + '</span><span class="badge"></span>';
      els.nodes.appendChild(el);
      n = { ...snap, el, badge, gone: false, pulse: false, x: 0, y: 0 };
      live.set(snap.key, n);
      if (instant) {
        el.classList.add("on");
      } else {
        const delay = (els.nodes.children.length - 1) * 70;
        setTimeout(() => { if (el.isConnected) el.classList.add("on"); }, delay);
      }
    } else {
      n.role = snap.role;
      n.parentKey = snap.parentKey;
      n.kind = snap.kind;
      n.gone = false;
      n.el.classList.remove("unmounting");
      n.el.classList.add("on");
    }
    n.badge = badge;
    const b = n.el.querySelector(".badge");
    const mark = { mount: "+", retain: "=", update: "~", unmount: "-" }[badge] ?? "";
    b.textContent = mark;
    b.className = "badge " + badge;
    return n;
  }

  function applyTopology(nodes, ops) {
    const nextKeys = new Set(nodes.map((n) => n.key));
    const opByKey = new Map((ops ?? []).map((o) => [o.key, o]));

    for (const snap of nodes) {
      const op = opByKey.get(snap.key);
      const badge = op ? op.op : live.has(snap.key) ? "retain" : "mount";
      ensureNode(snap, badge);
    }

    for (const [key, n] of live) {
      if (!nextKeys.has(key)) {
        n.gone = true;
        n.badge = "unmount";
        const b = n.el.querySelector(".badge");
        b.textContent = "-";
        b.className = "badge unmount";
        if (instant) {
          n.el.remove();
          live.delete(key);
        } else {
          n.el.classList.add("unmounting");
          n.el.classList.remove("on");
          const g = gen;
          setTimeout(() => {
            if (g !== gen) return;
            if (n.gone) {
              n.el.remove();
              live.delete(key);
              layoutLive();
            }
          }, 340);
        }
      }
    }

    const placed = center(layout(nodes));
    for (const p of placed) {
      const n = live.get(p.key);
      if (!n) continue;
      n.x = p.x;
      n.y = p.y;
      n.el.style.left = p.x + "px";
      n.el.style.top = p.y + "px";
    }
    renderEdges();
    renderPhys();

    if (ops && ops.length) {
      els.opsList.innerHTML = ops
        .map((o) => '<li class="' + o.op + '"><span class="m">' + o.mark + "</span> " + o.key + "</li>")
        .join("");
    }
  }

  function layoutLive() {
    const nodes = [...live.values()].filter((n) => !n.gone);
    if (!nodes.length) {
      renderEdges();
      renderPhys();
      return;
    }
    const placed = center(layout(nodes));
    for (const p of placed) {
      const n = live.get(p.key);
      if (!n) continue;
      n.x = p.x;
      n.y = p.y;
      n.el.style.left = p.x + "px";
      n.el.style.top = p.y + "px";
    }
    renderEdges();
    renderPhys();
  }

  function pulse(key) {
    if (instant) return;
    const n = live.get(key);
    if (!n) return;
    n.pulse = true;
    n.el.classList.add("pulse");
    renderEdges();
    setTimeout(() => {
      n.pulse = false;
      n.el.classList.remove("pulse");
      renderEdges();
    }, 420);
  }

  function formatTrace(ev) {
    const role = String(ev.role ?? "");
    const pad = role.padEnd(8, " ");
    if (role === "solve") {
      if (ev.phase === "v1") return "[" + pad + "] " + ev.output;
      return "[" + pad + "] " + ev.input;
    }
    if (role === "critic") {
      const start = String(ev.output).split(".")[0] ?? ev.output;
      return "[" + pad + "] " + start + "...";
    }
    return "[" + pad + "] " + ev.output;
  }

  function typeLine(text, cls) {
    const line = document.createElement("div");
    if (cls) line.className = cls;
    els.traceLog.appendChild(line);
    if (instant) {
      line.textContent = text;
    } else {
      let i = 0;
      const step = () => {
        i += 2;
        line.textContent = text.slice(0, i);
        if (i < text.length) setTimeout(step, 10);
      };
      step();
    }
    const kids = els.traceLog.children;
    while (kids.length > 8) kids[0].remove();
  }

  function setScore(value, label) {
    const num = Number(value);
    els.scoreVal.textContent = num.toFixed(2);
    els.scoreVal.className = "score" + (num >= 1 ? " one" : num <= 0 ? " zero" : "");
    els.scoreLabel.textContent = label ?? "";
  }

  function showAbstract(ev) {
    const authors = ev.authors ? '<span class="authors">' + ev.authors + "</span>" : "";
    els.abstract.innerHTML = '<span class="name">' + (ev.name ?? "") + authors + '</span><span class="body"></span>';
    els.abstract.classList.add("show");
    const body = els.abstract.querySelector(".body");
    const text = String(ev.abstract ?? "");
    if (instant) {
      body.textContent = text;
      return;
    }
    let i = 0;
    const step = () => {
      i += 3;
      body.textContent = text.slice(0, i);
      if (i < text.length && els.abstract.classList.contains("show")) {
        setTimeout(step, 12);
      }
    };
    step();
  }

  function applyEvent(ev) {
    switch (ev.type) {
      case "title":
        document.body.classList.add("cold");
        hideHeroOps();
        els.nowName.textContent = "vdom";
        els.nowTech.textContent = "virtual DOM for agents";
        break;
      case "paper":
        document.body.classList.remove("cold");
        hideHeroOps();
        els.nowName.textContent = ev.name;
        els.nowTech.textContent = ev.technique ?? "";
        els.stageMeta.textContent = ev.technique ?? "source";
        els.scientist.hidden = true;
        els.traceLog.textContent = "";
        els.graph.classList.add("reading");
        showAbstract(ev);
        break;
      case "compile":
        document.body.classList.remove("cold");
        els.nowName.textContent = ev.name;
        els.nowTech.textContent = "compile to AgentGraph";
        els.stageMeta.textContent = "compiler";
        break;
      case "mount":
        document.body.classList.remove("cold");
        hideHeroOps();
        els.abstract.classList.remove("show");
        els.scientist.hidden = true;
        els.graph.classList.remove("reading");
        els.nowName.textContent = ev.label ?? ev.graphId;
        els.stageMeta.textContent = "mount  " + ((ev.nodes && ev.nodes.length) || 0) + " nodes";
        applyTopology(ev.nodes ?? [], ev.ops ?? []);
        break;
      case "reconcile":
        document.body.classList.remove("cold");
        els.abstract.classList.remove("show");
        els.scientist.hidden = true;
        els.graph.classList.remove("reading");
        els.nowName.textContent = ev.label ?? (ev.from + " to " + ev.to);
        els.nowTech.textContent = "reconcile " + ev.from + " to " + ev.to;
        els.stageMeta.textContent = "diff desired vs current";
        els.traceLog.textContent = "";
        showHeroOps(ev.ops ?? []);
        applyTopology(ev.nodes ?? [], ev.ops ?? []);
        break;
      case "trace": {
        els.abstract.classList.remove("show");
        els.graph.classList.remove("reading");
        const line = formatTrace(ev);
        const out = String(ev.output ?? "");
        const cls = ev.role === "refine" || (ev.role === "actor" && out.indexOf("mod ") >= 0) ? "hit" : "";
        typeLine(line, cls);
        pulse(ev.nodeKey);
        break;
      }
      case "score":
        setScore(ev.value, ev.label);
        break;
      case "scientist":
        els.abstract.classList.remove("show");
        els.graph.classList.remove("reading");
        hideHeroOps();
        els.scientist.hidden = false;
        els.scientistText.textContent = ev.text ?? "emit replacement graph";
        els.nowName.textContent = "scientist";
        els.nowTech.textContent = ev.from + " to " + ev.to;
        els.stageMeta.textContent = "mutation";
        break;
      case "compare":
        els.scientist.hidden = true;
        els.graph.classList.remove("reading");
        hideHeroOps();
        els.compare.classList.add("hot");
        els.nowName.textContent = "vdom";
        els.nowTech.textContent = "reconcile a society that can rewrite itself";
        els.stageMeta.textContent = "Pi  DSH  vdom";
        break;
      default:
        break;
    }
  }

  function seek(to) {
    instant = true;
    try {
      resetVisual();
      if (!run) return;
      for (let i = 0; i < run.events.length; i++) {
        const ev = run.events[i];
        if (ev.t > to) {
          applied = i - 1;
          return;
        }
        applyEvent(ev);
      }
      applied = run.events.length - 1;
    } finally {
      instant = false;
    }
  }

  function tick(now) {
    if (!run) return;
    if (!last) last = now;
    const dt = now - last;
    last = now;
    if (playing) t += dt;
    if (t > run.duration) {
      t = 0;
      seek(0);
    }
    while (applied + 1 < run.events.length && run.events[applied + 1].t <= t) {
      applied += 1;
      applyEvent(run.events[applied]);
    }
    const pct = run.duration ? Math.min(1, t / run.duration) : 0;
    els.progress.style.width = (pct * 100) + "%";
    els.clock.textContent = (t / 1000).toFixed(1) + "s";
    requestAnimationFrame(tick);
  }

  function toggle() {
    playing = !playing;
    els.dot.classList.toggle("paused", !playing);
  }

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      toggle();
    }
    if (e.key === "r" || e.key === "R") {
      t = 0;
      seek(0);
    }
  });
  window.addEventListener("resize", () => layoutLive());

  fetch("run.json")
    .then((r) => {
      if (!r.ok) throw new Error("run.json missing");
      return r.json();
    })
    .then((doc) => {
      run = doc;
      const params = new URLSearchParams(location.search);
      const startAt = Number(params.get("t") || 0);
      const startPaused = params.get("pause") === "1";
      t = Number.isFinite(startAt) ? startAt : 0;
      if (startPaused) {
        playing = false;
        els.dot.classList.add("paused");
      }
      seek(t);
      requestAnimationFrame(tick);
    })
    .catch((err) => {
      els.nowName.textContent = "no run.json";
      els.nowTech.textContent = String(err.message ?? err);
    });
})();
