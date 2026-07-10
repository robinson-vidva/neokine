// neokine service worker — makes the app load instantly on repeat visits and
// work offline by caching the app shell and, crucially, the large MediaPipe
// library / WASM / model (which would otherwise be re-downloaded every visit).
//
// Strategy:
//   - cross-origin big assets (jsDelivr lib+WASM, Google-hosted model): cache-first
//     (they live at immutable, versioned URLs)
//   - same-origin navigations (index.html): network-first (fresh deploys win when
//     online; fall back to cache when offline)
//   - other same-origin files (css, js, icons, samples): stale-while-revalidate
//     (fast from cache, refreshed in the background)
//
// Bump VERSION to invalidate all caches on the next activation.
const VERSION = "neokine-v1";
const CORE = "core-" + VERSION; // same-origin shell
const CDN = "cdn-" + VERSION;   // cross-origin model/lib/wasm

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => !k.endsWith(VERSION)).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // Big, immutable third-party assets: cache-first (the whole point - don't
  // re-download the model + WASM every time).
  if (url.hostname.endsWith("jsdelivr.net") || url.hostname.endsWith("googleapis.com")) {
    e.respondWith(cacheFirst(req, CDN));
    return;
  }

  if (url.origin === self.location.origin) {
    if (req.mode === "navigate") { e.respondWith(networkFirst(req, CORE)); return; }
    e.respondWith(staleWhileRevalidate(req, CORE));
  }
  // anything else: let the browser handle it normally
});

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res && res.ok) cache.put(req, res.clone());
  return res;
}

async function networkFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(req);
    if (res && res.ok) cache.put(req, res.clone());
    return res;
  } catch (e) {
    return (await cache.match(req)) || (await cache.match("./")) ||
           (await cache.match("index.html")) || Response.error();
  }
}

async function staleWhileRevalidate(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  const fetching = fetch(req)
    .then((res) => { if (res && res.ok) cache.put(req, res.clone()); return res; })
    .catch(() => null);
  return hit || (await fetching) || Response.error();
}
