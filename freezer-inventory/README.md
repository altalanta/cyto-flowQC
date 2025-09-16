# Freezer Inventory (Local-first PWA)

A minimal, production-ish static web app for tracking a home freezer inventory. Works fully client-side with IndexedDB (via Dexie) and supports offline usage (PWA). Designed to be hosted on GitHub Pages.

## Features

- Add items with quantity + unit (g/kg, ml/L, count, serving)
- Inline creation of new foods during add
- Search + filters (location, opened, expiring in ≤ N days)
- Consume/decrement items with validation and density rules
- Minimal edit: location, opened, notes
- Export/Import JSON backup for migration between devices
- Offline-capable PWA with app-shell caching
- Accessible: semantic labels, keyboard typeahead, focus styles, aria-live toasts

## Tech

- Pure HTML/CSS/JavaScript — no build step, no frameworks
- Storage: IndexedDB via Dexie (loaded from CDN)
- PWA: `manifest.webmanifest` + `service-worker.js` (cache-first app shell)

## Run Locally

Option A (no SW):
- Open `index.html` directly in a browser. All features work except service worker install.

Option B (recommended): serve over HTTP so the SW can register.
- Python: `python3 -m http.server 5173` then open `http://localhost:5173/`
- Node: `npx http-server -p 5173`
- VS Code Live Server: Right‑click `index.html` → “Open with Live Server”

## Offline Check

- Open DevTools → Application → Service Workers → ensure it’s “activated”.
- Toggle “Offline” in DevTools and reload: app should load and data should persist.

## IndexedDB Check

- DevTools → Application → IndexedDB → inspect `freezer-db` → confirm `foods` and `items` after loading sample data.

## Export / Import

- Add a few items → click “Export JSON” → a file downloads.
- Hard refresh (or in a fresh browser) → click “Import JSON” and pick the file → items restored.

## Deploy to GitHub Pages

1. Create a new repository and push all files in this folder.
2. On GitHub: Settings → Pages → Source: “Deploy from branch”. Select branch `main`, folder `/` (root).
3. Wait for the Pages URL to appear. Open it.
4. Confirm the PWA install prompt is available (mobile Add to Home Screen or desktop Install).

## Manual Acceptance Tests

- Add: “Chicken thighs, 2 kg, Bin A, best-by 2025-12-15” → shows as “2 kg”.
- Decrement 300 g → shows “1.7 kg”.
- Try to decrement “100 ml” without density set → blocked with inline warning.
- Create new food inline: type “Pierogi” → “Create ‘Pierogi’” → submit → item appears.
- Filter by location; toggle “Opened only”; “Expiring in ≤ 7 days”.
- Export → Import into a fresh session → same items appear.

## Notes on Units & Density

- Stored in base units: grams (g), milliliters (ml), count, serving.
- Inputs accept kg/L and convert to base units; display upgrades back to kg/L when ≥ 1000.
- No automatic mass↔volume conversion unless the food has `densityGPerML`. If missing and a cross-type decrement is requested, the UI blocks with a warning.

## Files

- `index.html` — markup and structure
- `style.css` — simple, clean styling
- `db.js` — Dexie setup and data helpers
- `app.js` — UI logic and event handlers
- `manifest.webmanifest` — PWA manifest; icons include an SVG plus data-URL fallbacks
- `service-worker.js` — cache-first app shell; precaches Dexie CDN
- `icons/icon.svg` — simple scalable icon (fallback for PNGs)

## Accessibility

- Labels bound via `for`/`id`
- Keyboard support: Arrow keys + Enter in typeahead; Esc to close inline panels
- Toaster uses `aria-live=assertive`

## Limitations / Tips

- Because this is a static app, PNG icons are provided via SVG and data URLs in the manifest for portability. For best cross-browser install banners, you can replace them with real PNGs at `icons/icon-192.png` and `icons/icon-512.png` and update the manifest.
- Cross-origin Dexie CDN is precached by the service worker on first load; ensure you visit once online before going offline.

