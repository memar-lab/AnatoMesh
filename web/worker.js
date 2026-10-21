// Web Worker: the AnatoMesh engine, running on the visitor's own computer.
// Pyodide (CPython + NumPy + SciPy + scikit-image + matplotlib, compiled to WebAssembly) runs the desktop engine unchanged;
// py/webapi.py answers the interface's /api routes. Nothing is uploaded anywhere.
// (Pyodide 314 needs a module worker: new Worker(url, { type: "module" }); supported by Chrome, Edge, Safari 15+, Firefox 114+)
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

// Pyodide's own packages. nibabel, meshio and trimesh (pure Python, MIT) are not among them: they travel with the site in py/vendor.zip,
// so starting the engine never depends on PyPI.
const PYODIDE_PACKAGES = ["numpy", "scipy", "scikit-image", "matplotlib", "pandas", "packaging", "rich"];

let pyodide = null, webapi = null;
const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const status = (text) => post({ type: "status", text });
const memoryMB = () => { try { return Math.round(pyodide._module.HEAPU8.byteLength / 2 ** 20); } catch { return 0; } };   // WebAssembly memory reached so far (it never shrinks)

async function boot() {
  status("Downloading Python (Pyodide) ...");
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  status("Loading NumPy, SciPy, scikit-image, matplotlib ...");
  await pyodide.loadPackage(PYODIDE_PACKAGES);
  status("Unpacking nibabel, meshio, trimesh ...");
  const vendor = await fetch("../py/vendor.zip", { cache: "no-cache" });
  if (!vendor.ok) throw new Error("Cannot load py/vendor.zip (" + vendor.status + ")");
  pyodide.unpackArchive(await vendor.arrayBuffer(), "zip", { extractDir: "/anatomesh/vendor" });
  status("Loading the AnatoMesh engine ...");
  const manifest = await (await fetch("../py/manifest.json", { cache: "no-cache" })).json();
  const fetched = await Promise.all(manifest.files.map(async (f) => {
    const r = await fetch("../py/" + f, { cache: "no-cache" });
    if (!r.ok) throw new Error("Cannot load py/" + f + " (" + r.status + ")");
    return [f, new Uint8Array(await r.arrayBuffer())];
  }));
  for (const [f, bytes] of fetched) {
    const path = "/anatomesh/py/" + f;
    pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, bytes);
  }
  pyodide.FS.mkdirTree("/anatomesh/home");
  pyodide.FS.mkdirTree("/anatomesh/data");
  pyodide.runPython(`
import os, sys
os.environ["ANATOMESH_HOME"] = "/anatomesh/home"
os.environ["ANATOMESH_DATA"] = "/anatomesh/data"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, "/anatomesh/py")
sys.path.insert(1, "/anatomesh/vendor")
`);
  status("Starting the engine ...");
  webapi = pyodide.pyimport("webapi");
  post({ type: "ready", mem: memoryMB() });
}

let chain = Promise.resolve();                              // requests are answered one at a time, in order
self.onmessage = (ev) => { chain = chain.then(() => handle(ev.data)); };

async function handle(req) {
  const { id, method, path, query, body, upload } = req;
  try {
    if (!webapi) throw new Error("The engine is still starting.");
    let up = null;
    if (upload) {                                            // {file name: bytes} -> a real Python dict of bytes-like objects
      const obj = {};
      for (const [name, buf] of Object.entries(upload)) obj[name] = new Uint8Array(buf);
      up = pyodide.toPy(obj);
    }
    const proxy = webapi.call(method, path, JSON.stringify(query || {}), JSON.stringify(body === undefined ? null : body), up);
    if (up) up.destroy();
    const out = proxy.toJs();
    proxy.destroy();
    const meta = JSON.parse(out[0]);
    const bytes = out[1] instanceof Uint8Array ? out[1] : new Uint8Array(out[1]);
    const copy = bytes.slice().buffer;
    post({ type: "result", id, meta, body: copy, mem: memoryMB() }, [copy]);
  } catch (error) {
    const msg = JSON.stringify({ error: String(error.message || error) });
    const buf = new TextEncoder().encode(msg).buffer;
    post({ type: "result", id, meta: { status: 500, mime: "application/json" }, body: buf }, [buf]);
  }
  // a job has been queued by this request: run it now that the reply is on its way (the page follows it through posted updates)
  try {
    if (webapi && webapi.QUEUE.length > 0) webapi.run_queued();
  } catch (error) {
    post({ type: "log", text: "job failed: " + String(error.message || error) });
  }
}

boot().catch((error) => post({ type: "fatal", text: String(error.message || error) }));
