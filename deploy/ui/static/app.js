// HTD deploy monitor: one WebSocket, snapshots as JSON text, camera frames as binary JPEG.
(function () {
  "use strict";

  var STATES = ["BOOT", "STABILIZING", "READY", "RUNNING", "HOLD", "MANUAL", "ESTOP"];
  var CONTROL_HZ = 30;          // rollout's control rate: one predicted step per 1/30 s
  var T_MIN = -6, T_MAX = 2;    // plot time axis, s
  var STALE_S = 1.0;            // an input older than this is shown red
  // ...except chunks (one per 1.5 s: red after two missed) and commands (sent only on demand)
  var STALE_BY_TOPIC = { predicted_chunk: 3.0, ui_command: Infinity };
  var MISSING_S = 1e8;          // telemetry reports 1e9 for an input never received
  var LOG_MAX = 200;

  var PLOTS = [
    { title: "left hand", unit: "m", minSpan: 0.05,
      lines: [["left_hand_x", "x", "--s1"], ["left_hand_y", "y", "--s2"], ["left_hand_z", "z", "--s3"]] },
    { title: "right hand", unit: "m", minSpan: 0.05,
      lines: [["right_hand_x", "x", "--s1"], ["right_hand_y", "y", "--s2"], ["right_hand_z", "z", "--s3"]] },
    { title: "hand rotation error", unit: "deg", minSpan: 5,
      lines: [["left_rot_deg", "left", "--s1"], ["right_rot_deg", "right", "--s3"]] },
    { title: "fingers", unit: "mm", minSpan: 10,
      lines: [["left_fingers_mm", "left", "--s1"], ["right_fingers_mm", "right", "--s3"]] },
    { title: "torso height", unit: "m", minSpan: 0.05, lines: [["torso_height", "z", "--s4"]] },
    { title: "walk", unit: "m/s, rad/s", minSpan: 0.2,
      lines: [["vx", "vx", "--s2"], ["wz", "wz", "--s4"]] }
  ];

  var $ = function (id) { return document.getElementById(id); };
  var now = function () { return performance.now() / 1000; };

  var app = {
    ws: null, connected: false, retryMs: 500,
    snap: null,
    history: {},            // series -> [[t, v], ...], t on the page clock
    pred: null,             // {series, start, interval, horizon, received}
    camUrl: null, camAt: null,
    log: [], lastLogLine: null,
    drawPending: false,
    colors: {}
  };

  // ---- formatting -----------------------------------------------------------
  function fmt(v, digits) {
    return (typeof v === "number" && isFinite(v)) ? v.toFixed(digits) : "—";
  }
  function fmtAge(a) {
    if (a === null || a === undefined || !isFinite(a) || a >= MISSING_S) return "missing";
    return a < 1 ? (a * 1000).toFixed(0) + " ms" : a.toFixed(1) + " s";
  }
  function isStale(a, limit) {
    if (limit === undefined) limit = STALE_S;
    if (a === null || a === undefined || !isFinite(a)) return limit !== Infinity;
    return a > limit;
  }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // ---- connection -----------------------------------------------------------
  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    var ws = new WebSocket(proto + "//" + location.host + "/ws");
    ws.binaryType = "blob";
    app.ws = ws;
    ws.onopen = function () {
      app.connected = true; app.retryMs = 500;
      setConn(true);
      addLog("ui: connected", "local");
    };
    ws.onmessage = function (ev) {
      if (typeof ev.data === "string") onText(ev.data);
      else onCamera(ev.data);
    };
    ws.onclose = function () {
      if (app.connected) addLog("ui: disconnected, retrying", "err");
      app.connected = false;
      setConn(false);
      setTimeout(connect, app.retryMs);
      app.retryMs = Math.min(app.retryMs * 2, 5000);
    };
    ws.onerror = function () { ws.close(); };
  }

  function setConn(on) {
    var c = $("conn");
    c.textContent = on ? "connected" : "disconnected";
    c.className = "conn " + (on ? "on" : "off");
    renderButtons();
  }

  function onText(text) {
    var m;
    try { m = JSON.parse(text); } catch (e) { return; }
    if (m.type === "snapshot") onSnapshot(m);
    else if (m.type === "ack") addLog("ui: " + m.cmd + " sent (nonce " + m.nonce + ")", "local");
    else if (m.type === "error") addLog("ui: " + m.cmd + " rejected: " + m.error, "err");
  }

  function sendCommand(name) {
    if (app.ws && app.ws.readyState === WebSocket.OPEN) {
      app.ws.send(JSON.stringify({ cmd: name }));
      return;
    }
    fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ cmd: name }) })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.ok) addLog("ui: " + name + " sent over HTTP (nonce " + d.nonce + ")", "local");
        else addLog("ui: " + name + " rejected: " + d.error, "err");
      })
      .catch(function (e) { addLog("ui: " + name + " failed: " + e, "err"); });
  }

  function onSnapshot(s) {
    app.snap = s;
    var t = now();
    if (s.measured && !isStale(s.ages && s.ages.robot_state)) pushMeasured(s.measured, t);
    updatePrediction(s.predicted, t);
    renderState(s);
    renderRobot(s);
    renderCameraInfo(s);
    if (s.supervisor && s.supervisor.log_line && s.supervisor.log_line !== app.lastLogLine) {
      app.lastLogLine = s.supervisor.log_line;
      addLog(s.supervisor.log_line, "");
    }
    scheduleDraw();
  }

  // ---- state strip ----------------------------------------------------------
  function renderState(s) {
    var sup = s.supervisor;
    var state = sup ? sup.state : null;
    var boxes = document.querySelectorAll(".st");
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].classList.toggle("lit", boxes[i].getAttribute("data-state") === state);
    }
    $("state-card").classList.toggle("stale", isStale(s.ages && s.ages.supervisor));
    $("since").textContent = sup ? state + " for " + fmt(sup.since_s, 1) + " s" : "no supervisor status";
    $("reason").textContent = sup && sup.reason ? sup.reason : "";
    $("fault").textContent = sup && sup.fault ? "fault: " + sup.fault : "";

    var cond = $("conditions");
    cond.textContent = "";
    if (sup) {
      var items = [
        [sup.walker_live, "walker live"],
        [sup.standing, "standing " + fmt(sup.standing_for_s, 1) + " s"],
        [sup.policy_ready, "policy ready" + (s.policy && !s.policy.ready && s.policy.detail ? ": " + s.policy.detail : "")],
        [sup.frames_fresh, "policy frames"]
      ];
      items.forEach(function (it) { cond.appendChild(el("li", it[0] ? "ok" : "no", it[1])); });
    }
    var gating = $("gating");
    gating.textContent = "";
    (sup && sup.gating ? sup.gating : []).forEach(function (g) { gating.appendChild(el("li", "", g)); });

    setFlag($("walk"), "walk", sup ? sup.walk_enabled : null);
    setFlag($("manual-walk"), "manual walk", sup ? sup.manual_walk_enabled : null);
    renderButtons();
  }

  function setFlag(e, label, on) {
    e.textContent = label + ": " + (on === null ? "?" : on ? "on" : "off");
    e.className = "flag" + (on ? " on" : "");
  }

  function renderButtons() {
    var sup = app.snap && app.snap.supervisor;
    var ready = !!sup && sup.state === "READY" && !isStale(app.snap.ages && app.snap.ages.supervisor);
    $("btn-start").disabled = !(ready && app.connected);
  }

  function bindButtons() {
    $("btn-start").onclick = function () { sendCommand("start"); };
    $("btn-stop").onclick = function () { sendCommand("stop"); };
    $("btn-rearm").onclick = function () {
      if (confirm("Re-arm: leave HOLD / MANUAL, stand the robot and return to STABILIZING?")) sendCommand("rearm");
    };
    $("btn-reset").onclick = function () {
      if (confirm("Reset: end the session, reset the robot and return to STABILIZING?")) sendCommand("reset");
    };
  }

  // ---- robot panel ----------------------------------------------------------
  function renderRobot(s) {
    var m = s.measured || {}, sup = s.supervisor || {}, pol = s.policy || {};
    var deg = function (r) { return (typeof r === "number") ? r * 180 / Math.PI : null; };
    var rows = [
      ["torso height", fmt(m.torso_height, 3) + " m"],
      ["roll / pitch", fmt(deg(m.torso_roll), 1) + " / " + fmt(deg(m.torso_pitch), 1) + " deg"],
      ["walk applied", "vx " + fmt(m.vx, 2) + " m/s, wz " + fmt(m.wz, 2) + " rad/s"],
      ["manual walk", "vx " + fmt(sup.manual_vx, 2) + ", wz " + fmt(sup.manual_wz, 2)],
      ["controller tick", "p50 " + fmt(sup.tick_p50_ms, 2) + " / max " + fmt(sup.tick_max_ms, 2) + " ms"],
      ["rollout loop", "p50 " + fmt(pol.loop_p50_ms, 2) + " / max " + fmt(pol.loop_max_ms, 2) + " ms"],
      ["inference", fmt(pol.inference_ms, 1) + " ms, chunks " + (pol.chunks === undefined ? "—" : pol.chunks) +
        ", overrun " + (pol.overrun_steps === undefined ? "—" : pol.overrun_steps)],
      ["policy", s.policy ? (pol.ready ? "ready" : "not ready") +
        (pol.detail && pol.detail !== "ready" ? " (" + pol.detail + ")" : "") : "—"]
    ];
    var kv = $("robot-kv");
    kv.textContent = "";
    rows.forEach(function (r) { kv.appendChild(el("dt", "", r[0])); kv.appendChild(el("dd", "", r[1])); });

    drawStick($("stick-left"), sup.stick_lx, sup.stick_ly);
    drawStick($("stick-right"), sup.stick_rx, sup.stick_ry);
    $("buttons").textContent = sup.buttons && sup.buttons.length ? sup.buttons.join(" ") : "none";

    var ages = $("ages");
    ages.textContent = "";
    var list = [];
    Object.keys(s.ages || {}).forEach(function (k) { list.push([k, s.ages[k]]); });
    if (s.policy) {
      list.push(["rollout: robot state", pol.robot_state_age_s]);
      list.push(["rollout: hand state", pol.eef_state_age_s]);
      list.push(["rollout: camera", pol.camera_age_s]);
    }
    list.forEach(function (a) {
      var limit = STALE_BY_TOPIC[a[0]];
      var text = limit === Infinity && fmtAge(a[1]) === "missing" ? "none yet" : fmtAge(a[1]);
      ages.appendChild(el("li", isStale(a[1], limit) ? "stale" : "", a[0] + " " + text));
    });
  }

  function drawStick(cv, x, y) {
    var ctx = cv.getContext("2d"), w = cv.width, h = cv.height, c = app.colors;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = c["--border"]; ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    ctx.beginPath();
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();
    if (typeof x !== "number" || typeof y !== "number") return;
    var px = w / 2 + Math.max(-1, Math.min(1, x)) * (w / 2 - 5);
    var py = h / 2 - Math.max(-1, Math.min(1, y)) * (h / 2 - 5);
    ctx.fillStyle = c["--lit"];
    ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();
  }

  // ---- camera ---------------------------------------------------------------
  function onCamera(blob) {
    var url = URL.createObjectURL(blob.type ? blob : new Blob([blob], { type: "image/jpeg" }));
    var img = $("camera");
    var prev = app.camUrl;
    img.src = url;
    app.camUrl = url;
    app.camAt = now();
    if (prev) URL.revokeObjectURL(prev);
  }

  function renderCameraInfo(s) {
    var age = s.ages ? s.ages.camera : null;
    var info = $("camera-info");
    info.textContent = fmtAge(age);
    info.style.color = isStale(age) ? "var(--bad)" : "";
  }

  // ---- plots ----------------------------------------------------------------
  function pushMeasured(m, t) {
    PLOTS.forEach(function (p) {
      p.lines.forEach(function (l) {
        var key = l[0], v = m[key];
        var buf = app.history[key] || (app.history[key] = []);
        buf.push([t, (typeof v === "number") ? v : NaN]);
        while (buf.length && buf[0][0] < t + T_MIN - 0.5) buf.shift();
      });
    });
  }

  function updatePrediction(p, t) {
    if (!p) { app.pred = null; return; }
    if (app.pred && app.pred.received === p.received) return;
    // Step k is drawn at arrival + k / 30 s; `age` is the server-side time since arrival.
    app.pred = { series: p, start: t - (p.age || 0), interval: p.interval, horizon: p.horizon, received: p.received };
  }

  function buildPlots() {
    var root = $("plots");
    PLOTS.forEach(function (p) {
      var box = el("div", "plot");
      var head = el("div", "plot-head");
      head.appendChild(el("b", "", p.title));
      head.appendChild(el("span", "muted", p.unit));
      p.legend = [];
      p.lines.forEach(function (l) {
        var s = el("span", "");
        var sw = el("span", "swatch");
        sw.style.background = "var(" + l[2] + ")";
        s.appendChild(sw);
        var txt = el("span", "mono", l[1]);
        s.appendChild(txt);
        head.appendChild(s);
        p.legend.push(txt);
      });
      p.canvas = el("canvas");
      box.appendChild(head);
      box.appendChild(p.canvas);
      root.appendChild(box);
    });
  }

  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    ["--fg", "--muted", "--border", "--grid", "--lit", "--future", "--mark", "--s1", "--s2", "--s3", "--s4"]
      .forEach(function (k) { app.colors[k] = cs.getPropertyValue(k).trim(); });
  }

  function scheduleDraw() {
    if (app.drawPending) return;
    app.drawPending = true;
    requestAnimationFrame(function () { app.drawPending = false; drawPlots(); });
  }

  function drawPlots() {
    var t = now();
    PLOTS.forEach(function (p) { drawPlot(p, t); });
  }

  function predPoints(key, t) {
    var pr = app.pred;
    if (!pr || !pr.series[key]) return [];
    return pr.series[key].map(function (v, k) { return [pr.start + k / CONTROL_HZ - t, v]; });
  }

  function yRange(p, t) {
    var lo = Infinity, hi = -Infinity;
    p.lines.forEach(function (l) {
      (app.history[l[0]] || []).forEach(function (q) {
        var v = q[1];
        if (q[0] - t >= T_MIN && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      });
      predPoints(l[0], t).forEach(function (q) {
        if (q[0] <= T_MAX && isNum(q[1])) { lo = Math.min(lo, q[1]); hi = Math.max(hi, q[1]); }
      });
    });
    if (!isFinite(lo)) { lo = -p.minSpan / 2; hi = p.minSpan / 2; }
    var mid = (lo + hi) / 2, span = Math.max(hi - lo, p.minSpan) * 1.15;
    return [mid - span / 2, mid + span / 2];
  }

  function drawPlot(p, t) {
    var cv = p.canvas, dpr = window.devicePixelRatio || 1;
    var cw = cv.clientWidth, ch = cv.clientHeight;
    if (!cw || !ch) return;
    if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) {
      cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
    }
    var ctx = cv.getContext("2d"), c = app.colors;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    var L = 42, R = 6, T = 6, B = 18, w = cw - L - R, h = ch - T - B;
    var yr = yRange(p, t);
    var X = function (s) { return L + (s - T_MIN) / (T_MAX - T_MIN) * w; };
    var Y = function (v) { return T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * h; };

    // future region, grid, axes labels
    ctx.fillStyle = c["--future"];
    ctx.fillRect(X(0), T, X(T_MAX) - X(0), h);
    ctx.strokeStyle = c["--grid"]; ctx.lineWidth = 1;
    ctx.fillStyle = c["--muted"]; ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "top";
    for (var s = T_MIN; s <= T_MAX; s += 1) {
      ctx.beginPath(); ctx.moveTo(X(s), T); ctx.lineTo(X(s), T + h); ctx.stroke();
      ctx.textAlign = s === T_MIN ? "left" : s === T_MAX ? "right" : "center";
      if (s % 2 === 0) ctx.fillText((s > 0 ? "+" : "") + s + " s", X(s), T + h + 3);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (var i = 0; i <= 3; i++) {
      var v = yr[0] + (yr[1] - yr[0]) * i / 3;
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + w, Y(v)); ctx.stroke();
      ctx.fillText(v.toPrecision(3), L - 4, Y(v));
    }
    ctx.strokeStyle = c["--muted"];
    ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), T + h); ctx.stroke();

    // where rollout switches to the next chunk
    if (app.pred && app.pred.interval) {
      var xm = X(app.pred.start + app.pred.interval / CONTROL_HZ - t);
      if (xm >= L && xm <= L + w) {
        ctx.strokeStyle = c["--mark"]; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(xm, T); ctx.lineTo(xm, T + h); ctx.stroke();
      }
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, w, h); ctx.clip();
    p.lines.forEach(function (l, j) {
      var col = c[l[2]];
      var hist = (app.history[l[0]] || []).map(function (q) { return [q[0] - t, q[1]]; });
      strokeLine(ctx, hist, X, Y, col, []);
      strokeLine(ctx, predPoints(l[0], t), X, Y, col, [5, 4]);
      var last = hist.length ? hist[hist.length - 1][1] : NaN;
      p.legend[j].textContent = l[1] + " " + (isNum(last) ? last.toPrecision(3) : "—");
    });
    ctx.restore();
  }

  function strokeLine(ctx, pts, X, Y, color, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash);
    ctx.beginPath();
    var pen = false;
    pts.forEach(function (q) {
      if (!isNum(q[1])) { pen = false; return; }
      if (pen) ctx.lineTo(X(q[0]), Y(q[1])); else ctx.moveTo(X(q[0]), Y(q[1]));
      pen = true;
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ---- log ------------------------------------------------------------------
  function addLog(text, cls) {
    var d = new Date();
    var stamp = [d.getHours(), d.getMinutes(), d.getSeconds()].map(function (n) { return n < 10 ? "0" + n : "" + n; }).join(":");
    var last = app.log[app.log.length - 1];
    if (last && last.text === text) return;       // collapse consecutive repeats
    app.log.push({ text: text, cls: cls, stamp: stamp });
    if (app.log.length > LOG_MAX) app.log.shift();
    renderLog();
  }

  function renderLog() {
    var box = $("log"), filter = $("log-filter").value.trim().toLowerCase();
    var atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    box.textContent = "";
    app.log.forEach(function (e) {
      if (filter && e.text.toLowerCase().indexOf(filter) < 0) return;
      var li = el("li", e.cls);
      li.appendChild(el("span", "t", e.stamp));
      li.appendChild(document.createTextNode(e.text));
      box.appendChild(li);
    });
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  // ---- start ----------------------------------------------------------------
  function init() {
    readColors();
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var onScheme = function () { readColors(); scheduleDraw(); };
      if (mq.addEventListener) mq.addEventListener("change", onScheme); else if (mq.addListener) mq.addListener(onScheme);
    }
    buildPlots();
    bindButtons();
    $("log-filter").addEventListener("input", renderLog);
    window.addEventListener("resize", scheduleDraw);
    connect();
  }

  init();
})();
