// Browser edition of AnatoMesh: the unchanged interface (assets/) believes it talks to the local engine over HTTP (/api/...).
// This script, loaded before the interface, answers those calls itself with the engine running in a Web Worker (web/worker.js)
// on the visitor's own computer. No request leaves the page except the one-time download of Python (jsDelivr) and of the
// pure-Python libraries (PyPI); inputs and results stay on the user's machine.
(() => {
  "use strict";
  const SELF = document.currentScript ? document.currentScript.src : location.href;   // where this file lives (worker.js is beside it)
  const realFetch = window.fetch.bind(window);

  // ------------------------------------------------------------------------------------------------ engine (worker)
  let worker = null, nextId = 1, bootState = "starting", resolveReady, ready;
  const pending = new Map();
  const jobs = new Map();                                   // job id -> payload of GET /api/jobs/<id> (kept up to date by the engine)
  let dirHandle = null, pickerDeclined = false, engineMB = 0;                // engineMB: WebAssembly memory of the engine, as last reported

  const newReady = () => { ready = new Promise((r) => (resolveReady = r)); };
  newReady();

  function startEngine() {
    bootState = "starting";
    showOverlay("Starting the engine on your computer ...");
    worker = new Worker(new URL("worker.js", SELF), { type: "module" });
    worker.onmessage = (ev) => {
      const d = ev.data;
      if (typeof d === "string") {                         // posted by Python (webapi.py): job status / log
        let m; try { m = JSON.parse(d); } catch { return; }
        if (m.type === "job") jobs.set(m.id, m.payload);
        return;
      }
      if (d.type === "status") { showOverlay(d.text); setPill(d.text, "busy"); }
      else if (d.type === "ready") { bootState = "ready"; hideOverlay(); setPill("Engine ready: running on this computer", "ok"); resolveReady(); }
      else if (d.type === "fatal") { bootState = "failed"; showOverlay("The engine could not start: " + d.text, true); setPill("Engine failed to start", "bad"); }
      else if (d.type === "result") { const p = pending.get(d.id); pending.delete(d.id); if (d.mem) engineMB = d.mem; if (p) p(d); }
    };
    worker.onerror = (e) => { setPill("Engine error", "bad"); console.error(e); };
  }

  function engine(method, path, query, body, upload) {
    return new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      const transfer = upload ? Object.values(upload) : [];
      worker.postMessage({ id, method, path, query, body, upload }, transfer);
    });
  }

  async function restartEngine(why) {
    try { worker.terminate(); } catch { /* already gone */ }
    for (const [, resolve] of pending) resolve({ meta: { status: 500, mime: "application/json" }, body: new TextEncoder().encode('{"error":"engine restarted"}').buffer });
    pending.clear();
    for (const [id, j] of jobs) if (j.state !== "cancelled") jobs.delete(id);        // their files lived in the stopped engine
    newReady();
    startEngine();
    showOverlay(why + " Restarting the engine ...");
  }

  // ------------------------------------------------------------------------------------------------ fetch override
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
  const fail = (msg, status = 400) => json({ error: msg }, status);

  window.fetch = async function (input, init) {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const at = url.pathname.indexOf("/api/");
    if (at < 0) return realFetch(input, init);
    const path = url.pathname.slice(at);                      // /api/...
    const method = ((init && init.method) || (typeof input !== "string" && input.method) || "GET").toUpperCase();
    const seg = path.split("/").filter(Boolean).slice(1);

    // answered here, without the engine (it may be busy computing)
    if (method === "GET" && path === "/api/health")
      return json({ ok: true, version: "web", home: "browser", python: "Pyodide", tools: { ants: false, synthseg: false }, web: true });
    if (method === "GET" && path === "/api/system") {
      return json({ cpu: 0, ram: Math.min(100, Math.round((100 * engineMB) / 2048)), gpu: null });   // share of the ~2 GB a tab can use
    }
    await ready;
    if (method === "GET" && path === "/api/jobs") {
      const kind = url.searchParams.get("kind");
      const items = [...jobs.values()].map(stripJob).filter((i) => !kind || i.kind === kind).sort((a, b) => (b.created || 0) - (a.created || 0));
      return json({ jobs: items });
    }
    if (method === "GET" && seg[0] === "jobs" && seg.length === 2 && jobs.has(seg[1])) return json(jobs.get(seg[1]));
    if (method === "POST" && seg[0] === "jobs" && seg[2] === "cancel") {
      const j = jobs.get(seg[1]);
      if (!j) return fail("job not found", 404);
      if (j.state === "running" || j.state === "queued") {
        j.state = "cancelled"; j.finished = Date.now() / 1000;
        await restartEngine("The job was cancelled. Cancelling stops the whole engine, so earlier results are cleared.");
      }
      return json(stripJob(j));
    }
    if (method === "DELETE" && seg[0] === "jobs" && seg.length === 2) {
      const j = jobs.get(seg[1]);
      if (j && (j.state === "running" || j.state === "queued")) {
        await restartEngine("The running job was removed.");
        jobs.delete(seg[1]);
        return json({ ok: true });
      }
      const r = await engine("DELETE", path, {}, undefined, null);
      if (r.meta.status < 400) jobs.delete(seg[1]);
      return respond(r);
    }
    if (method === "POST" && seg[0] === "jobs" && seg[2] === "export") return exportJob(seg[1], path, init);

    // everything else goes to the engine
    const query = Object.fromEntries(url.searchParams.entries());
    let body, upload = null;
    const raw = init && init.body;
    if (raw instanceof FormData) {
      upload = {};
      for (const [k, v] of raw.entries()) {
        if (v instanceof Blob) upload[v.name || "input"] = await v.arrayBuffer();
        else query[k] = String(v);
      }
    } else if (typeof raw === "string") {
      try { body = JSON.parse(raw); } catch { body = undefined; }
    }
    const r = await engine(method, path, query, body, upload);
    if (r.meta.status < 400 && method === "POST" && (seg[0] === "prepare" || seg[0] === "runs")) {
      const id = JSON.parse(new TextDecoder().decode(r.body)).id;          // show the job at once; the engine then keeps it up to date
      if (id && !jobs.has(id)) jobs.set(id, { id, kind: seg[0] === "prepare" ? "prepare" : "mesh", state: "queued", progress: 0, stage: 0, stage_name: "",
                                              message: "", created: Date.now() / 1000, log: [], job: {} });
    }
    return respond(r);
  };

  const stripJob = (j) => { const { job, log, trace, ...item } = j; return item; };
  const respond = (r) => new Response(r.body, { status: r.meta.status, headers: { "Content-Type": r.meta.mime || "application/octet-stream", "Cache-Control": "no-store" } });

  // ------------------------------------------------------------------------------------------------ export -> the user's own folder
  async function exportJob(jobId, path, init) {
    // the folder chooser needs the click that started the export, so it is the first thing that happens
    if (!dirHandle && !pickerDeclined && window.showDirectoryPicker) {
      try { dirHandle = await window.showDirectoryPicker({ mode: "readwrite", id: "anatomesh-out" }); setFolderLabel(); }
      catch { pickerDeclined = true; }                        // cancelled: fall back to one .zip download
    }
    let body; try { body = JSON.parse(init.body); } catch { body = {}; }
    const r = await engine("POST", path, {}, body, null);
    if (r.meta.status >= 400 || r.meta.mime !== "files") return respond(r);
    const files = {};
    let off = 0;
    r.meta.names.forEach((name, i) => { files[name] = r.body.slice(off, off + r.meta.sizes[i]); off += r.meta.sizes[i]; });
    const saved = await saveFiles(files, "AnatoMesh_" + jobId);
    return json({ written: saved });
  }

  async function saveFiles(files, stem) {
    const names = Object.keys(files);
    if (dirHandle) {
      try {
        if ((await dirHandle.requestPermission({ mode: "readwrite" })) === "granted") {
          const sub = await dirHandle.getDirectoryHandle(stem, { create: true });
          for (const n of names) {
            const h = await sub.getFileHandle(n.replace(/[\\/]/g, "_"), { create: true });
            const w = await h.createWritable(); await w.write(files[n]); await w.close();
          }
          return names.map((n) => dirHandle.name + "/" + stem + "/" + n);
        }
      } catch (e) { console.warn("folder write failed; downloading instead", e); }
    }
    download(stem + ".zip", makeZip(files));
    return names.map((n) => "Downloads/" + stem + ".zip :: " + n);
  }

  function download(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
  }

  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  const crc32 = (u8) => { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  function makeZip(files) {                                   // store-only zip, no dependency
    const enc = new TextEncoder(), parts = [], central = [];
    let offset = 0;
    for (const [name, raw] of Object.entries(files)) {
      const data = new Uint8Array(raw), nm = enc.encode(name), crc = crc32(data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint32(14, crc, true);
      h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, nm.length, true);
      parts.push(new Uint8Array(h.buffer), nm, data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint32(16, crc, true);
      c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, nm.length, true); c.setUint32(42, offset, true);
      central.push(new Uint8Array(c.buffer), nm);
      offset += 30 + nm.length + data.length;
    }
    const size = central.reduce((s, p) => s + p.length, 0), end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, central.length / 2, true); end.setUint16(10, central.length / 2, true);
    end.setUint32(12, size, true); end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
  }

  // ------------------------------------------------------------------------------------------------ a small status bar + start-up overlay
  const css = `
  #am-bar{position:fixed;left:10px;bottom:10px;z-index:99998;display:flex;gap:8px;align-items:center;font:12px/1.3 system-ui,sans-serif;
    background:rgba(20,24,32,.92);color:#e6e9ee;border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:5px 6px 5px 12px;max-width:calc(100vw - 20px)}
  #am-bar .pill{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #am-bar .dot{width:8px;height:8px;border-radius:50%;background:#f5b942;flex:none}#am-bar .ok .dot{background:#34b27b}#am-bar .bad .dot{background:#f97066}
  #am-bar button{font:inherit;border:1px solid rgba(255,255,255,.25);background:transparent;color:inherit;border-radius:999px;padding:3px 10px;cursor:pointer}
  #am-bar button:hover{border-color:#7aa2f7}
  #am-overlay{position:fixed;inset:0;z-index:99999;background:rgba(10,12,16,.82);color:#e6e9ee;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font:15px/1.5 system-ui,sans-serif;text-align:center;padding:24px}
  #am-overlay .spin{width:40px;height:40px;border:4px solid rgba(255,255,255,.25);border-top-color:#7aa2f7;border-radius:50%;animation:am-spin 1s linear infinite}
  #am-overlay[hidden]{display:none}#am-overlay.err .spin{display:none}#am-overlay small{opacity:.7;max-width:520px}
  @keyframes am-spin{to{transform:rotate(360deg)}}
  #am-info{position:fixed;inset:0;z-index:99999;background:rgba(10,12,16,.6);display:flex;align-items:center;justify-content:center;padding:16px}
  #am-info>div{background:#1b2027;color:#e6e9ee;border:1px solid #313845;border-radius:12px;max-width:620px;max-height:85vh;overflow:auto;padding:18px 22px;font:14px/1.55 system-ui,sans-serif}
  #am-info h3{margin:0 0 8px;font-size:17px}#am-info li{margin:4px 0}#am-info button{margin-top:10px;font:inherit;padding:6px 14px;border-radius:8px;border:1px solid #5b8def;background:#2b4a8c;color:#fff;cursor:pointer}`;
  const style = document.createElement("style"); style.textContent = css; document.head.appendChild(style);

  const bar = document.createElement("div"); bar.id = "am-bar";
  bar.innerHTML = '<span class="pill busy"><i class="dot"></i><span id="am-pill-text">starting engine ...</span></span>' +
    '<button id="am-folder" title="Chrome / Edge: write exports straight into a folder on your computer">Output folder</button>' +
    '<button id="am-about">About</button>';
  const overlay = document.createElement("div"); overlay.id = "am-overlay"; overlay.hidden = true;
  overlay.innerHTML = '<div class="spin"></div><div id="am-overlay-text"></div><small>The first start downloads Python and its libraries (about 60-90 MB) to your browser; later visits use the browser cache. Your files are never uploaded.</small>';
  document.addEventListener("DOMContentLoaded", () => {
    document.body.append(bar, overlay);
    if (!window.showDirectoryPicker) bar.querySelector("#am-folder").hidden = true;      // phones / Firefox / Safari: exports download as a .zip
    startEngine();
  });

  function setPill(text, kind) { const p = bar.querySelector(".pill"); p.className = "pill " + kind; bar.querySelector("#am-pill-text").textContent = text; }
  function showOverlay(text, isError) { overlay.hidden = false; overlay.classList.toggle("err", !!isError); overlay.querySelector("#am-overlay-text").textContent = text; }
  function hideOverlay() { overlay.hidden = true; }
  function setFolderLabel() { bar.querySelector("#am-folder").textContent = "Output folder: " + (dirHandle ? dirHandle.name : "downloads"); }

  document.addEventListener("click", async (e) => {
    if (e.target.id === "am-folder") {
      if (!window.showDirectoryPicker) { alert("This browser cannot write into a folder directly (use Chrome or Edge). Exports are downloaded as one .zip file."); return; }
      try { dirHandle = await window.showDirectoryPicker({ mode: "readwrite", id: "anatomesh-out" }); pickerDeclined = false; setFolderLabel(); } catch { /* cancelled */ }
    }
    if (e.target.id === "am-about") {
      const box = document.createElement("div"); box.id = "am-info";
      box.innerHTML = `<div><h3>AnatoMesh, browser edition</h3><ul>
        <li><b>Runs on your computer.</b> The page you opened is a set of static files; Python runs in your browser (WebAssembly). Your files are never uploaded.</li>
        <li><b>Input:</b> a label map (NIfTI, .nii or .nii.gz) with tissue codes 1-8 or FreeSurfer aseg / aparc codes. Raw MRI segmentation (needs ANTs registration), diffusion fibres, arteries, Harvard-Oxford / registration-made atlases and FreeSurfer folders need the desktop version.</li>
        <li><b>Size:</b> a browser tab has about 2 GB of memory. Use an element size of 3 mm or coarser for a whole head; finer meshes belong on the desktop version.</li>
        <li><b>Speed:</b> single-threaded, without the compiled kernels of the desktop version, so it is slower.</li>
        <li><b>Results:</b> Export writes into the folder you choose (Chrome / Edge) or downloads one .zip.</li>
        <li><b>Cancel</b> stops the whole engine and clears earlier results of this session.</li></ul>
        <button id="am-close">Close</button></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (ev) => { if (ev.target === box || ev.target.id === "am-close") box.remove(); });
    }
  });
  window.__anatomeshWeb = { engine, jobs, ready: () => ready, state: () => bootState, memoryMB: () => engineMB };
})();
