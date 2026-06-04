// A custom element that renders a Spine 4.3 animation with chine-web
// (chine compiled to WebAssembly + a WebGL2 renderer).
//
// Usage:
//   <script type="module" src="chine-web/js/chine-spine.js"></script>
//   <chine-spine atlas="skeleton.atlas" skeleton="skeleton.skel"
//                animation="idle" style="width:600px;height:600px"></chine-spine>
//
// `skeleton` may be a binary `.skel` or a `.json` export (chosen by extension).
// The element registers as both <chine-spine> and, for drop-in compatibility
// with Spine's HTML export, <spine-skeleton> (unless one is already defined).

import init, { WebSpine } from "../pkg/chine_web.js";

let wasmReady = null;
function ensureWasm(wasmUrl) {
  if (!wasmReady) wasmReady = init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  return wasmReady;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to load image: " + url));
    img.src = url;
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

    const skelUrl = this.getAttribute("skeleton");
    const atlasUrl = this.getAttribute("atlas");
    if (!skelUrl || !atlasUrl) {
      throw new Error("chine-spine needs `skeleton` and `atlas` attributes");
    }
    const base = atlasUrl.slice(0, atlasUrl.lastIndexOf("/") + 1);

    this._resize();
    const atlasText = await fetch(atlasUrl).then((r) => r.text());

    let spine;
    if (skelUrl.endsWith(".json")) {
      const json = await fetch(skelUrl).then((r) => r.text());
      spine = WebSpine.from_json(this._canvas, json, atlasText);
    } else {
      const bytes = new Uint8Array(await fetch(skelUrl).then((r) => r.arrayBuffer()));
      spine = WebSpine.from_binary(this._canvas, bytes, atlasText);
    }
    this._spine = spine;

    // Upload each atlas page image as a texture, in page order.
    for (const name of spine.page_names()) {
      spine.add_page(await loadImage(base + name));
    }

    const anim = this.getAttribute("animation") || spine.animation_names()[0];
    if (anim) spine.set_animation(anim, true);

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
