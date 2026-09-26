// Minimal static file server for the chine-web demos, with the correct MIME
// types for ES modules and WebAssembly. Dev helper only.
//
//   node chine-web/scripts/serve.js chine-web 8090
const http = require("http");
const fs = require("fs");
const path = require("path");

const root = path.resolve(process.argv[2] || ".");
const port = Number(process.argv[3] || 8090);
const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".png": "image/png",
  ".atlas": "text/plain",
  ".skel": "application/octet-stream",
  ".css": "text/css",
};

function reply(res, status, text) {
  res.writeHead(status, { "Content-Type": "text/plain" });
  res.end(text);
}

http
  .createServer((req, res) => {
    let rel;
    try {
      rel = decodeURIComponent(req.url.split("?")[0]);
    } catch {
      // A malformed percent escape would otherwise throw and stop the server.
      reply(res, 400, "400 bad request");
      return;
    }
    const fp = path.join(root, rel === "/" ? "/index.html" : rel);
    // Serve only files under the root: `..` segments must not climb out.
    if (!fp.startsWith(root + path.sep)) {
      reply(res, 403, "403 " + rel);
      return;
    }
    fs.readFile(fp, (err, data) => {
      if (err) {
        reply(res, 404, "404 " + rel);
        return;
      }
      res.writeHead(200, {
        "Content-Type": MIME[path.extname(fp).toLowerCase()] || "application/octet-stream",
        "Access-Control-Allow-Origin": "*",
      });
      res.end(data);
    });
  })
  .listen(port, () => console.log(`serving ${root} on http://localhost:${port}`));
