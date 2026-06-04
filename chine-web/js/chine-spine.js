// A custom element that renders a Spine 4.3 animation with chine-web
// (chine compiled to WebAssembly + a WebGL2 renderer).
//
// Usage by URL:
//   <script type="module" src="chine-web/js/chine-spine.js"></script>
//   <chine-spine atlas="skeleton.atlas" skeleton="skeleton.skel"
//                animation="idle" style="width:600px;height:600px"></chine-spine>
//
// Drop-in for a Spine HTML export: swap the spine-webgl runtime <script> for
// this module. The element reads the export's embedded base64 globals
// (`skeletonData`, `atlasData`, `textureData`) and registers under Spine's
// element name `<spine-skeleton>` as well as `<chine-spine>`. The skeleton may
// be a binary `.skel` or a `.json` export; the type is auto-detected.

import init, { WebSpine } from "../pkg/chine_web.js";

let wasmReady = null;
function ensureWasm(wasmUrl) {
  if (!wasmReady) wasmReady = init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  return wasmReady;
}

function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeUtf8(bytes) {
  return new TextDecoder("utf-8").decode(bytes);
}

// A binary `.skel` starts with arbitrary hash bytes; a JSON export starts with
// `{` (after optional whitespace).
function looksLikeJson(bytes) {
  for (let i = 0; i < Math.min(bytes.length, 16); i++) {
    const c = bytes[i];
    if (c === 0x7b) return true;
    if (c !== 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d) return false;
  }
  return false;
}

function makeSpine(canvas, skelBytes, atlasText) {
  return looksLikeJson(skelBytes)
    ? WebSpine.from_json(canvas, decodeUtf8(skelBytes), atlasText)
    : WebSpine.from_binary(canvas, skelBytes, atlasText);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to load image: " + src));
    img.src = src;
  });
}

class ChineSpine extends HTMLElement {
  connectedCallback() {
    if (this._booted) return;
    this._booted = true;
    this._canvas = document.createElement("canvas");
    this._canvas.style.cssText = "display:block;width:100%;height:100%";
    this.appendChild(this._canvas);
    this._boot().catch((err) => console.error("[chine-spine]", err));
  }

  disconnectedCallback() {
    this._stopped = true;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  async _boot() {
    await ensureWasm(this.getAttribute("wasm") || undefined);
    this._resize();

    // Drop-in mode: a Spine HTML export embeds its data as base64 globals.
    const g = globalThis;
    const embedded =
      typeof g.skeletonData !== "undefined" && typeof g.atlasData !== "undefined";

    let textures = null;
    let base = "";
    if (embedded) {
      const skelBytes = base64ToBytes(g.skeletonData);
      const atlasText = decodeUtf8(base64ToBytes(g.atlasData));
      this._spine = makeSpine(this._canvas, skelBytes, atlasText);
      textures = new Map();
      if (Array.isArray(g.textureData)) {
        for (const [name, b64] of g.textureData) textures.set(name, b64);
      }
    } else {
      const skelUrl = this.getAttribute("skeleton");
      const atlasUrl = this.getAttribute("atlas");
      if (!skelUrl || !atlasUrl) {
        throw new Error("chine-spine needs `skeleton` and `atlas` attributes or embedded data");
      }
      base = atlasUrl.slice(0, atlasUrl.lastIndexOf("/") + 1);
      const atlasText = await fetch(atlasUrl).then((r) => r.text());
      const skelBytes = new Uint8Array(await fetch(skelUrl).then((r) => r.arrayBuffer()));
      this._spine = makeSpine(this._canvas, skelBytes, atlasText);
    }

    // Upload each atlas page image as a texture, in page order.
    for (const name of this._spine.page_names()) {
      const embeddedPng = textures && textures.get(name);
      const src = embeddedPng ? "data:image/png;base64," + embeddedPng : base + name;
      this._spine.add_page(await loadImage(src));
    }

    const anim = this.getAttribute("animation") || this._spine.animation_names()[0];
    if (anim) this._spine.set_animation(anim, true);

    this._last = performance.now();
    this._loop();
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this._canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this._canvas.clientHeight * dpr));
    if (this._canvas.width !== w) this._canvas.width = w;
    if (this._canvas.height !== h) this._canvas.height = h;
    this._w = w;
    this._h = h;
  }

  _loop() {
    if (this._stopped || !this._spine) return;
    const now = performance.now();
    const dt = Math.min((now - this._last) / 1000, 0.064);
    this._last = now;
    this._resize();
    this._spine.frame(dt, this._w, this._h);
    this._raf = requestAnimationFrame(() => this._loop());
  }
}

customElements.define("chine-spine", ChineSpine);
// Drop-in for existing Spine HTML exports: register under Spine's element name
// too, unless the page already provides it.
if (!customElements.get("spine-skeleton")) {
  customElements.define("spine-skeleton", class extends ChineSpine {});
}

export { ChineSpine };
