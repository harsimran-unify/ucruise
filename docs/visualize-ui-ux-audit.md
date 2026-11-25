# UI/UX Audit — `localTesting/visualize.html`

**Date:** 2026-11-24 · **Scope:** full-file review (pre-fix revision, 5,445 lines) + complete fix/re-theme pass.
**Method:** manual review of every CSS rule, HTML block, and JS function, cross-checked against the `ui-ux-pro-max` skill library (Minimalism & Swiss Style guidance + zinc monochrome palette + its priority categories: accessibility → touch → performance → style → layout → typography → animation → forms → navigation → charts).
**Outcome:** file rewritten in place — all functional bugs fixed, dead UI removed, theme changed to a light minimalist monochrome (Swiss) design. Line numbers below reference the **pre-fix** revision.

---

## Summary

| Severity | Found | Resolution |
|---|---|---|
| Critical (broken functionality) | 5 | Fixed |
| High (bugs with user impact) | 14 | Fixed |
| Medium (UX flow & interaction) | 12 | Fixed |
| Accessibility | 10 | Fixed |
| Visual / theme | 10 | Superseded by re-theme |
| Performance | 6 | Fixed |
| Dead UI | 6 features | Removed |

---

## 1 · Critical — broken functionality

| # | Finding (old line refs) | Resolution |
|---|---|---|
| C1 | **Route Editor entirely non-functional** — 13 functions called but never written (`prepareEditorData` 4972, `loadRoutesIntoEditor` 4973/5230/5267, `initializeEditorState` 2871/2960, `saveRouteChanges` 2193, `revertRouteChanges` 2196, `undoEdit`/`redoEdit` 2213–2218, `recordHistoryState` 4980, `addToHistory` 5231/5270, `renderHistoryTimeline` 5042, `updateSimulator` 5044, `applySimulatedChanges` 2259, `searchUnassignedUsers` 2176). Clicking *Edit Routes* threw a `ReferenceError` and half-initialized a dead panel. | Editor + History + What-If removed entirely (user decision). |
| C2 | **`updateMapHeaderControls` never defined** but called on every render (3541) and edit-mode toggle (5006) → `ReferenceError` after every render tick. | Calls removed with edit mode. |
| C3 | **Wrong utilization in Route Details** (3336): fraction displayed as `0.8%` instead of `80%`; donut arc near-zero (3378); efficiency rating compared fraction vs 80/50 → *always* "Needs Improvement" (3484). | Fixed: percentage math + consistent 80/50 thresholds. |
| C4 | **Stale details cache** (3325): `routeDetailsCache` keyed by index only, never invalidated → panel showed the *previous algorithm's* driver/users after a mode switch. | Cache removed; panel re-renders (cheap). |
| C5 | **Ghost routes after re-render**: Directions callbacks (4043) drew polylines after `clearMap()` when data was re-rendered mid-flight — orphan hit-areas that could never be cleared. | `renderGeneration` counter; stale callbacks drop out. |

## 2 · High — bugs with user impact

| # | Finding | Resolution |
|---|---|---|
| H1 | Zoom-adaptive hover hit areas never worked — listener registered at parse time when `map` was still null (3998). | Listener registered inside `initMap`. |
| H2 | `routeLayers` never cleared → unbounded growth + stale zoom adaptation (3843, 3989). | Cleared in `clearMap`. |
| H3 | Duplicate `highlightRoute` definitions (4680 vs 5358) — second silently overwrote the first. | Both removed (palette click → details; drag flow gone with editor). |
| H4 | Filters didn't compose & stomped state: clearing search reloaded `/routes` and reset the algorithm mode (2958); utilization (4636) and route dropdown (4613) each re-derived from base ignoring the other filters. | One composed pipeline (`activeSearch` + `activeMinUtil` + `activeRouteFilter` → `applyFilters()`); no refetch on filter changes. |
| H5 | Colors/indices shifted after filtering — `colorForRoute(index, filteredLength)` re-mapped every route's color on each search (2300, 3561); palette/dropdown indices diverged from map indices. | Colors and numbering derive from the **full-dataset index**; filters produce a `visible` view. |
| H6 | Driver click opened an InfoWindow *and* a floating HTML popup that never hid (`hideRouteInfo` 4534 never called) and whose "Occupancy Trend" sparkline was **fabricated `Math.random()` data** re-rolled per hover (4540). | Floating popup + sparkline removed; InfoWindow only. |
| H7 | `updateMarkerIcon` degraded the glass marker to a plain circle after any drag (5386). | Moot — dragging removed with editor. |
| H8 | Drag thresholds inconsistent: drop-zone highlight at <5 km (5199) but drop accepted only <3 km (5120) → highlighted target then rejected. | Moot — removed with editor. |
| H9 | `switchEditorTab` matched tabs by label text — "simulator" never matched the "What-If" label (5030) → tab never highlighted. | Moot — removed. |
| H10 | Run-all copy stale ("all 5 algorithms" — 9 modes run, 2883); blocking overlay for 1–3 min with no elapsed feedback or dismissal. | Accurate copy, live elapsed timer, click-to-dismiss (run continues). |
| H11 | Every display toggle re-rendered the map → fresh billable Google Directions calls per route per toggle (4724–4752); >8-stop routes silently fell back to straight lines (Directions waypoint limit). | Directions results cached per (mode, route, stops); >20 stops short-circuits with a one-time notice; Directions failures surface a warning toast instead of console-only. |
| H12 | `adjustBrightness` swapped green/blue channels (4390–4393) — wrong gradient hues everywhere. | Channel math fixed. |
| H13 | Okabe-ito palette slot 8 was `#000000` — invisible black routes on the dark map (2286). | Replaced with Okabe gray `#999999` (also correct on the new light map). |
| H14 | XSS via `innerHTML`: `first_name`, `email`, `address`, driver/vehicle fields interpolated unescaped into panels, info windows, alerts (3408–3470, 3800–3817). | `escapeHtml()` applied to every data-driven interpolation. |

## 3 · Medium — UX flow & interaction

| # | Finding | Resolution |
|---|---|---|
| M1 | Map drew `optimizeWaypoints: true` (4037) — Google silently **re-ordered pickups**, misrepresenting the algorithm's actual sequence. | `optimizeWaypoints: false`; the line shows the planned stop order. |
| M2 | Hardcoded run parameters (`ASSIGN_PARAMS` 2275–2280) locked the dashboard to UC_logisticllp / Evening shift. | Editable Source ID / Parameter / Shift / Choice inputs (prefilled) in the Algorithm section. |
| M3 | CSV export had no quoting/escaping — commas in names broke rows (4871–4874); silently exported only the filtered subset. | Proper escaping; exports the full dataset with a count confirmation. |
| M4 | "Generate Report" button showed a **PDF icon** but downloaded JSON (2035, 4887). | Renamed "Export JSON Report" with `fa-file-code`. |
| M5 | "Capture Map" button existed only to tell you it doesn't work (2037, 4935). | Removed. |
| M6 | "Route Thumbnails" opened a "coming soon" panel (2062, 4848). | Removed. |
| M7 | Alerts accumulated forever (`showAlert` 4451 never pruned); system alerts wiped them on the next analytics tick. | Capped at 6, success/warning auto-expire after 6 s, `role="alert"` on errors. |
| M8 | No empty state — a filter matching nothing just showed a blank map. | Empty-dataset warning; stats drop to 0 visibly. |
| M9 | Infinite spinner if Google Maps failed to load (no error path; `initMap` never ran). | `onerror` hook + `gm_authFailure` hook + 8 s timeout → explicit error message in the overlay. |
| M10 | `directionsService` was an implicit global (2600) — would throw under strict mode. | Properly declared. |
| M11 | "Active Drivers" stat = route count, including filtered routes (4212). | Distinct `driver_id` count over the visible set. |
| M12 | Palette "All" + dropdown + hover + click were three overlapping ways to do the same thing with inconsistent side effects (dropdown value set without firing its filter). | Palette = inspect (details + spotlight), dropdown = filter, hover = preview — each with one clear effect. |

## 4 · Accessibility

| # | Finding | Resolution |
|---|---|---|
| A1 | Invisible focus indicator: 0.14-alpha `:focus-within` glow (3027); `outline: none` on inputs. | Global 2 px `:focus-visible` ring (#18181B, 2 px offset); inputs get a solid black focus border. |
| A2 | Cluster palette swatches were `div onclick` — not keyboard operable (3010–3039). | Real `<button type="button">` elements with `aria-label`s. |
| A3 | Details panel had no dialog semantics, no Esc, no focus management (2129). | `role="dialog"`, Esc closes, focus moves to close button on open and returns to the trigger on close. |
| A4 | Icon-only buttons without labels: details close "×" (2135), collapse button. | `aria-label` on both; collapse button also gets `aria-expanded`. |
| A5 | Toggle buttons conveyed state by color alone. | `aria-pressed` on all six display toggles + label swap. |
| A6 | 8–9 px text (inspector footer 1597, metric labels 1547/1565, badge 1512). | Inspector removed; smallest text now ≥10 px, body 15 px, labels ≥11 px. |
| A7 | No `prefers-reduced-motion` support; pulse/bounce/shine animations always on. | Global reduced-motion media query. |
| A8 | Emoji as icons everywhere (markers, legend, buttons, info windows — 🚗👤🏢🛣️✏️💡). | Replaced with Font Awesome in HTML and drawn SVG glyphs inside marker data-URIs (per skill: "emoji as icons" anti-pattern). |
| A9 | Utilization slider had no `aria-valuetext`. | Set on init and on input (`aria-valuetext="45%"`), step 5. |
| A10 | Hover-only affordances (route spotlight appeared only on hover after 2 s). | Kept, but click/hover now mirror each other and the legend tip documents the interaction; palette offers a keyboard path to the same details. |

## 5 · Visual / theme (superseded by re-theme)

The original dark glassmorphism had systemic issues, the worst being **scrambled design tokens**: `--primary-500: #242425` (near-black) was used for section icons (278), metric values (1772), section titles (1677), range thumb (334) and focus borders (307) — i.e. **invisible ink on dark backgrounds** — while the real indigo was hardcoded inline in half the components (377, 660, 912, 1609…). Related: three conflicting `#map` rules (542/685), dead grid rules for a layout that no longer existed (969–1018), double scrollbars (sidebar *and* sidebar-content scrolled, 1021 vs 238), solid slate panels breaking the glass metaphor (301/493/1762), a 4 px resize handle, sub-30 px touch targets, unbounded legend growth, and a ✓ overlapping the number on active swatches (913).

**Resolution:** the entire stylesheet was replaced. New design per the skill's "Minimalism & Swiss Style" row + its zinc palette:

- **Light paper theme**: `#FAFAFA` canvas, `#FFFFFF` surfaces, zinc text scale (`#09090B` → `#A1A1AA`), `#E4E4E7` hairline borders, **no** shadows/gradients/blur/glow, sharp corners.
- Single "accent" = black; **red reserved for errors only** (alerts, fatal overlay).
- Grayscale Google Maps base (POI/transit hidden); the Labels toggle flips only map labels.
- Data encodings stay colorblind-safe (Okabe-ito routes/markers, monochrome heatmap with red hot-end) — color carries *data*, never decoration.
- Touch targets ≥36–38 px; 12 px resize-handle hit area with hairline hover cue; legend capped with internal scroll; active swatch swaps its number for a check.

## 6 · Performance

| # | Finding | Resolution |
|---|---|---|
| P1 | Every marker SVG carried `feGaussianBlur`/`feMerge` filters (3572–3586, 3638–3676) — hundreds of per-frame filter compositions. | Flat single-pass SVGs (gradient-free, filter-free). |
| P2 | Marker DROP animation replayed for every marker on *every* re-render. | Staggered (~40 ms, capped 600 ms), first paint of a dataset only. |
| P3 | `backdrop-filter: blur(20px) saturate(180%)` on ~10 stacked surfaces. | Gone with the theme (flat surfaces). |
| P4 | Re-render storms from filter/toggle interactions (see H11). | Directions cache + composed single-pass filtering. |
| P5 | `transform: translateY(-2px)` on hover **and** focus moved buttons around (650–657). | Hover = background tint only; focus = outline only (no motion). |
| P6 | Buttons/panels used `transition: all` (118, 351, 627). | Narrow transitions (background/border/color, 150–250 ms). |

## 7 · Dead UI removed (user decision)

Route Editor panel + tabs (Users & Routes / History / What-If), edit-mode header state, marker drag-and-drop machinery, mini-maps panel + button, Capture Map button, floating route-info popup + fake sparkline, `routeVisibility`/`isolateRoute`/`focusRoute` stubs, dead keyframes (`markerBounce`/`markerDrop`/`markerPulse` — never referenced), `.route-dimmed/.route-highlighted` CSS (never referenced), and the 13 phantom editor functions + `updateMapHeaderControls`.

## Follow-ups

1. **Server copy**: `localTesting/` is gitignored — sync `visualize.html` to the server copy manually when deploying (per CLAUDE.md).
2. **Maps key**: the Google Maps JS key is still hardcoded in the script URL (kept for local use; it's a referrer-restricted client key by design). Worth moving to a server-injected template variable and confirming the key's HTTP-referrer restrictions in Cloud Console.
3. **Not changed on purpose**: `local_server.py`, the algorithms, and `run_and_view.py` are untouched; nothing was committed (per standing instruction).
