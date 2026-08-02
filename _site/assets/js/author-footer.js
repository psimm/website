(() => {
  const mount = document.getElementById("author-footer");
  if (!mount) return;

  const normalize = (path) => {
    try {
      const url = new URL(path, window.location.origin);
      let pathname = url.pathname.replace(/\/+$/, "");
      if (pathname.endsWith("/index.html")) {
        pathname = pathname.slice(0, -"/index.html".length);
      } else if (pathname.endsWith(".html")) {
        pathname = pathname.slice(0, -".html".length);
      }
      return pathname || "/";
    } catch {
      return path;
    }
  };

  const rootRelative = (path) => {
    const normalized = path.startsWith("/") ? path : `/${path}`;
    return normalized.replace(/\/+/g, "/");
  };

  const initNav = () => {
    const nav = document.getElementById("blog-post-nav");
    if (!nav) return;

    const current = normalize(window.location.pathname);

    Promise.all([
      fetch("/listings.json").then((response) => response.json()),
      fetch("/search.json").then((response) => response.json()),
    ])
      .then(([listings, search]) => {
        const blogListing = listings.find((entry) => entry.listing === "/blog.html");
        if (!blogListing?.items?.length) return;

        const titles = new Map();
        for (const item of search) {
          if (!item.href || item.section) continue;
          if (!item.href.startsWith("blog/") || !item.href.includes("/index.html")) {
            continue;
          }
          titles.set(normalize(item.href), item.title);
        }

        // listings.json is newest-first; Previous = older, Next = newer
        const items = blogListing.items.map((item) => ({
          href: rootRelative(item),
          path: normalize(item),
          title: titles.get(normalize(item)) || item,
        }));

        const index = items.findIndex((item) => item.path === current);
        if (index === -1) return;

        const older = items[index + 1];
        const newer = items[index - 1];
        let visible = false;

        if (older) {
          const prev = document.getElementById("blog-post-nav-prev");
          const link = prev.querySelector(".blog-post-nav-link");
          link.href = older.href;
          link.textContent = older.title;
          prev.hidden = false;
          visible = true;
        }

        if (newer) {
          const next = document.getElementById("blog-post-nav-next");
          const link = next.querySelector(".blog-post-nav-link");
          link.href = newer.href;
          link.textContent = newer.title;
          next.hidden = false;
          visible = true;
        }

        if (visible) nav.hidden = false;
      })
      .catch(() => {
        // Leave the author section; skip navigation if listing data is unavailable.
      });
  };

  fetch("/assets/author-footer.html")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    })
    .then((html) => {
      mount.outerHTML = html;
      initNav();
    })
    .catch(() => {
      // Keep the empty mount if the fragment cannot be loaded.
    });
})();
