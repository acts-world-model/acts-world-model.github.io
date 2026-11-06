// Renders the video sections from window.ACTS_VIDEOS (static/js/videos.js).
(function () {
  "use strict";

  const data = window.ACTS_VIDEOS;
  const RATES = [0.25, 0.5, 1];
  let playbackRate = 0.5;

  // Lazy-load and autoplay videos while they are on screen.
  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        const v = e.target;
        if (e.isIntersecting) {
          if (!v.src && v.dataset.src) v.src = v.dataset.src;
          v.playbackRate = playbackRate;
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      }
    },
    { rootMargin: "200px 0px" }
  );

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }

  // Lazy-loaded; autoplays while on screen.
  function video(src, aspect) {
    const v = el("video");
    if (aspect) v.style.aspectRatio = aspect;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.controls = true;
    v.preload = "none";
    v.dataset.src = src;
    v.addEventListener("loadedmetadata", () => { v.playbackRate = playbackRate; });
    observer.observe(v);
    return v;
  }

  // `caption` is text or a node.
  function clip(src, caption, aspect) {
    const f = el("figure", "clip");
    const v = video(src, aspect);
    f.appendChild(v);
    if (caption) {
      const c = el("figcaption");
      c.append(caption);
      f.appendChild(c);
    }
    return { node: f, video: v };
  }

  // Calls fn(fraction) while `v` plays and whenever it seeks.
  function follow(v, fn) {
    const tick = () => {
      if (v.duration) fn(v.currentTime / v.duration);
      if (!v.paused) requestAnimationFrame(tick);
    };
    v.addEventListener("play", () => requestAnimationFrame(tick));
    v.addEventListener("seeked", tick);
  }

  // Input / prediction bar under a clip. The playhead follows the video; click to seek.
  function timeline(v, spec) {
    const pred = spec.total - spec.input;
    const bar = el("div", "timeline");
    bar.title = "Click to seek";
    const inp = el("div", "tl-seg tl-input");
    inp.style.flex = String(spec.input);
    const out = el("div", "tl-seg tl-pred");
    out.style.flex = String(pred);
    if (spec.block) {
      out.classList.add("blocks");
      out.style.setProperty("--blocks", String(pred / spec.block));
    }
    const head = el("div", "tl-head");
    bar.append(inp, out, head);

    const legend = el("div", "tl-legend");
    const inLabel = el("span", "tl-key tl-key-input", `Input: ${spec.input} recorded frames`);
    const outLabel = el("span", "tl-key tl-key-pred", spec.block
      ? `Prediction: ${pred} frames, generated ${spec.block} at a time`
      : `Prediction: ${pred} frames`);
    legend.append(inLabel, outLabel);

    follow(v, (p) => {
      head.style.left = (p * 100).toFixed(2) + "%";
      const inInput = p * spec.total < spec.input;
      [inp, inLabel].forEach((n) => n.classList.toggle("active", inInput));
      [out, outLabel].forEach((n) => n.classList.toggle("active", !inInput));
    });
    bar.addEventListener("click", (e) => {
      const r = bar.getBoundingClientRect();
      if (v.duration) v.currentTime = ((e.clientX - r.left) / r.width) * v.duration * 0.999;
    });
    const wrap = el("div", "tl");
    wrap.append(bar, legend);
    return wrap;
  }

  function speedControl() {
    const box = el("span", "speed");
    box.setAttribute("aria-label", "Playback speed");
    RATES.forEach((r) => {
      const b = el("button", null, r + "×");
      b.type = "button";
      b.dataset.rate = String(r);
      b.setAttribute("aria-pressed", String(r === playbackRate));
      b.addEventListener("click", () => {
        playbackRate = r;
        document.querySelectorAll(".speed button").forEach((o) =>
          o.setAttribute("aria-pressed", String(o.dataset.rate === String(r))));
        document.querySelectorAll("video").forEach((v) => { v.playbackRate = r; });
      });
      box.appendChild(b);
    });
    return box;
  }

  function buttonRow(cls, labels, onSelect) {
    const row = el("div", cls);
    row.setAttribute("role", "tablist");
    const buttons = labels.map((label, i) => {
      const b = el("button", "tab", label);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => onSelect(i));
      row.appendChild(b);
      return b;
    });
    row.mark = (i) => buttons.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    return row;
  }

  // Object buttons + clip buttons + one stage that shows the selected clip.
  function player(container, groups, render, label) {
    const tasks = data.tasks.filter((t) => (groups[t.id] || []).length);
    if (!tasks.length) return;
    const bar = el("div", "player-bar");
    const stage = el("div", "stage");
    let clipRow = null;
    const objects = buttonRow("tabs objects", tasks.map((t) => t.name), selectTask);
    bar.append(objects, speedControl());
    container.append(bar, stage);

    function show(items, i) {
      clipRow.mark(i);
      stage.querySelectorAll("video").forEach((v) => observer.unobserve(v));
      stage.replaceChildren(render(items[i]));
    }
    function selectTask(t) {
      objects.mark(t);
      const items = groups[tasks[t].id];
      const next = buttonRow("tabs clips", items.map((it, i) => label(it, i)), (i) => show(items, i));
      if (clipRow) clipRow.replaceWith(next); else bar.after(next);
      clipRow = next;
      show(items, 0);
    }
    selectTask(0);
  }

  const clipLabel = (it, i) => String(i + 1);

  // Failure clips name their failure mode above the one-sentence caption.
  function caption(it) {
    if (!it.mode) return it.caption;
    const box = el("span", "failure-note");
    box.append(el("strong", "failure-mode", it.mode), " ", it.caption);
    return box;
  }

  function single(spec) {
    return (it) => {
      const c = clip(it.src, caption(it), it.aspect);
      const wrap = el("div");
      wrap.append(c.node, timeline(c.video, spec));
      return wrap;
    };
  }

  // One clip with a shared ground-truth row on top and one row per method below it.
  function comparison(spec) {
    return (g) => {
      const c = clip(g.src, "Rows, top to bottom: " + ["Ground truth"].concat(g.rows).join(", ") + ".", "1280 / 864");
      const wrap = el("div", "compare");
      wrap.append(c.node, timeline(c.video, spec));
      if (g.force) wrap.appendChild(forceChart(g.force, c.video));
      return wrap;
    };
  }

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, text) {
    const n = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text) n.textContent = text;
    return n;
  }

  // Recorded normal force over the 16-frame window; the playhead follows `video`.
  function forceChart(force, video) {
    const W = 900, H = 160, L = 44, R = 22, T = 16, B = 34;
    const n = force.left.length;
    const peak = Math.max(...force.left, ...force.right, 0.5);
    const yMax = Math.ceil(peak * 1.15);
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (f) => T + (1 - f / yMax) * (H - T - B);

    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "force-chart", role: "img",
      "aria-label": "Recorded normal force for the left and right sensors" });
    const split = x((n / 2) - 0.5);
    s.appendChild(svg("rect", { x: L, y: T, width: split - L, height: H - T - B, class: "fc-history" }));
    s.appendChild(svg("text", { x: (L + split) / 2, y: T + 13, class: "fc-note" }, "input"));
    s.appendChild(svg("text", { x: (split + W - R) / 2, y: T + 13, class: "fc-note" }, "prediction"));
    [0, yMax / 2, yMax].forEach((v) => {
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "fc-grid" }));
      s.appendChild(svg("text", { x: L - 8, y: y(v) + 4, class: "fc-tick", "text-anchor": "end" }, String(+v.toFixed(1))));
    });
    [[0, "0 s"], [(n - 1) / 2, "0.5 s"], [n - 1, "1.0 s"]].forEach(([i, label]) =>
      s.appendChild(svg("text", { x: x(i), y: H - B + 18, class: "fc-tick", "text-anchor": "middle" }, label)));
    s.appendChild(svg("text", { x: 12, y: (T + H - B) / 2, class: "fc-tick", "text-anchor": "middle",
      transform: `rotate(-90 12 ${(T + H - B) / 2})` }, "force (N)"));
    [["left", "fc-left"], ["right", "fc-right"]].forEach(([k, cls]) => {
      const pts = force[k].map((f, i) => `${x(i).toFixed(1)},${y(f).toFixed(1)}`).join(" ");
      s.appendChild(svg("polyline", { points: pts, class: cls }));
    });
    const head = svg("line", { x1: L, x2: L, y1: T, y2: H - B, class: "fc-head" });
    s.appendChild(head);
    follow(video, (p) => {
      const px = L + p * (W - L - R);
      head.setAttribute("x1", px);
      head.setAttribute("x2", px);
    });

    const f = el("figure", "fig plot");
    f.appendChild(s);
    const cap = el("figcaption");
    cap.innerHTML = '<span class="key fc-left-key"></span>left sensor &nbsp; ' +
      '<span class="key fc-right-key"></span>right sensor &nbsp; Recorded normal force.';
    f.appendChild(cap);
    return f;
  }

  const SHORT = { input: 8, total: 16 };
  const LONG = { input: 8, total: 128, block: 8 };

  document.addEventListener("DOMContentLoaded", () => {
    const at = (id) => document.getElementById(id);
    player(at("short-videos"), data.shortHorizon, single(SHORT), clipLabel);
    player(at("robot-videos"), data.robot, single(SHORT), clipLabel);
    player(at("ablation-videos"), data.ablations, comparison(SHORT), (it) => it.label);
    player(at("failure-videos"), data.failures, single(SHORT), clipLabel);
    player(at("long-videos"), data.longHorizon, single(LONG), clipLabel);
    if (at("baseline-videos")) player(at("baseline-videos"), data.baselines, comparison(SHORT), (it) => it.label);
  });
})();
