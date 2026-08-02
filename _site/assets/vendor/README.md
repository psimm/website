# Vendored frontend dependencies

These files are served from `simmering.dev` so visiting the site does not contact public JavaScript or CSS CDNs. Versions are pinned deliberately.

| File | Upstream source | License |
|---|---|---|
| `jquery/jquery-3.5.1.min.js` | [jQuery 3.5.1](https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js) | MIT |
| `requirejs/require-2.3.6.min.js` | [RequireJS 2.3.6](https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js) | MIT |
| `plotly/plotly-3.3.0.min.js` | [Plotly.js 3.3.0](https://cdn.plot.ly/plotly-3.3.0.min.js) | MIT |
| `plotly/plotly-2.26.0.min.js` | [Plotly.js 2.26.0](https://cdn.plot.ly/plotly-2.26.0.min.js) | MIT |
| `datatables/jquery.dataTables-1.12.1.mjs` | [DataTables 1.12.1](https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs) | MIT |
| `datatables/jquery.dataTables-1.13.1.min.css` | [DataTables 1.13.1](https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css) | MIT |
| `mathjax/tex-svg-full-3.2.2.js` | [MathJax 3.2.2](https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js) | Apache-2.0 |
| `goatcounter/count.js` | [GoatCounter client](https://gc.zgo.at/count.js) | ISC |

`scripts/localize_remote_assets.py` rewrites frozen notebook output after each Quarto render and fails the build if a remote frontend library remains.
