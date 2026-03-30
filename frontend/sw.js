/* Nimbus PWA — cache shell only; API/WebDAV stay network-only. */
const CACHE_NAME = "nimbus-shell-v2";
const PRECACHE_URLS = [
  "/",
  "/assets/manifest.webmanifest",
  "/assets/logo.svg",
  "/assets/logo-banner.svg",
  "/assets/icons/icon-192.png",
  "/assets/icons/icon-512.png",
];

function isApiOrDynamic(pathname) {
  if (pathname === "/health") return true;
  if (pathname.startsWith("/search")) return true;
  if (pathname.startsWith("/photo")) return true;
  if (pathname.startsWith("/index")) return true;
  if (pathname.startsWith("/tags")) return true;
  return false;
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        Promise.allSettled(PRECACHE_URLS.map((u) => cache.add(u))),
      )
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== CACHE_NAME)
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  if (url.pathname === "/sw.js") {
    event.respondWith(fetch(req));
    return;
  }

  if (isApiOrDynamic(url.pathname)) {
    event.respondWith(fetch(req));
    return;
  }

  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200 && res.type === "basic") {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
        }
        return res;
      })
      .catch(() =>
        caches.match(req).then((cached) => {
          if (cached) return cached;
          if (url.pathname === "/" || url.pathname === "/index.html") {
            return caches.match("/");
          }
          return Promise.reject(new Error("offline"));
        }),
      ),
  );
});
