# Endid.jl — project notes for Claude

Distributional difference-in-differences in Julia. Implements Lee &
Wooldridge (2025): transform the panel using pre-treatment outcomes
(demean or detrend each unit against its own pre-period), collapse to a
cross section, then fit `Engression.jl` on that cross section to recover
the full counterfactual outcome distribution — not just a mean. From the
fitted model you get both an ATT and quantile treatment effects (QTE),
which is the point of using engression instead of plain OLS/TWFE: it's
useful when treatment shifts spread or tails, not just the mean.

## Relationship to other packages in this workspace

- **Depends on `../Engression.jl`** (declared via `[sources]` in
  `Project.toml`, path = `../Engression.jl`, sibling directory). This is
  a hard requirement — `Pkg.instantiate()` will fail if that directory
  isn't present next to this one. Engression.jl is the actual
  distributional-regression / generative-model engine; Endid.jl only
  does the panel bookkeeping (transform, cross-section construction,
  bootstrap) and calls `engression()` / `predict()` / `sample()`.
- **`~/projects/software/torch-endid`** is a GPU-accelerated Python port
  of this same methodology (companion to
  `~/projects/software/torch-engression`, the Python port of
  Engression.jl). Same math, different runtime: this repo is Julia/CPU,
  torch-endid is Python/PyTorch/GPU. If you're asked to change the
  method itself (transform, ATT/QTE construction, bootstrap logic), check
  whether the same fix is needed on the torch side — they were built to
  match, and they can drift out of sync silently. Don't assume this
  Julia package is the only implementation.

## What's where

- `src/Endid.jl` — the entire package (~410 lines), single file:
  - `apply_transform` — per-unit demean/detrend against pre-period outcomes,
    tags each unit's first post-treatment row as `firstpost`.
  - `fit_engression_cs` — fits `engression(X, Y)` on the transformed cross
    section (X = treatment dummy + optional controls), then computes ATT as
    mean of counterfactual `predict(..., target="mean")` differences, and
    QTE as differences in quantiles of pooled `sample()` draws (treated vs.
    counterfactual-untreated).
  - `endid(df, y, id, time, post; ...)` — common-timing DiD entry point.
  - `endid_staggered(df, y, id, time, gvar; ...)` — staggered-adoption
    version: loops over treatment cohorts (each compared against
    never-treated units only), then pools cohort ATT/QTE bootstrap draws
    with treated-unit-count weights.
  - `EndidResult` — struct holding `att`, `se`, `ci`, `qte` (DataFrame),
    `model` (the underlying `Engressor`), `design`. Has a `Base.show`
    method and a `RecipesBase` plot recipe (QTE curve + 95% ribbon + ATT
    line).
- `test/runtests.jl` — one `@testset`, 7 `@test`s: transform sanity checks
  + one full `endid()` run on a synthetic panel with a known ATT=2.0,
  checked with tiny `num_epochs`/`nboot` for speed.
- `vignettes/` — source `.qmd` (Quarto) for the two docs vignettes;
  `docs/src/vignettes/*.md` are the rendered-into-Documenter copies.
- `docs/` — Documenter.jl site, deployed via `.github/workflows/docs.yml`
  on push to `master`. `docs/build/` is committed output (gh-pages via
  Pages artifact, not a separate branch).

## Tests

```
cd ~/projects/software/Endid.jl
julia --project=. test/runtests.jl
```

Takes ~1–1.5 minutes (most of it is one real engression fit + a 5-rep
bootstrap, each spawning its own fit). Last verified: **7/7 pass**
(2026-07-01).

## Manifest.toml is intentionally committed here

Unlike some sibling packages in this workspace (where a committed
`Manifest.toml` caused Julia-version conflicts in multi-version CI
matrices), Endid.jl's only CI is `docs.yml` — a single Julia version,
docs-build only, no test matrix. So there's no multi-version conflict
risk, and `Manifest.toml` is committed for reproducible instantiation
(a fresh clone without it would fail `Pkg.instantiate()` — this was an
actual bug, fixed 2026-07-01 in commit `0b41db3`). Don't delete it
reflexively just because "packages shouldn't commit Manifest.toml" is
the right default elsewhere — here it's deliberate. Do regenerate it
(`Pkg.instantiate()` then commit) if `Project.toml` deps change.

## Gotchas found while reading the code

- **Bootstrap parallelism is asymmetric between the two entry points.**
  `endid()`'s bootstrap loop uses `Threads.@threads` (parallel across
  `nboot` reps). `endid_staggered()`'s per-cohort bootstrap loop is a
  plain serial `for b in 1:nboot` — it is NOT threaded. If you're
  running staggered designs with many cohorts × many bootstrap reps and
  it's slow, that loop is the place to parallelize
  (`Threads.@threads`, consistent with the project-wide MC-parallelism
  convention) — it currently isn't.
- **Never-treated units are required for `endid_staggered`.** It errors
  out (`"No never-treated units found."`) if every unit in `gvar` is
  eventually treated — there is no not-yet-treated comparison group
  option here, only never-treated.
- **Cohorts can be silently dropped.** If a cohort's cross-section after
  dropping missings has `nrow(cs) < 4` or fewer than 2 treated/2 control
  units, `endid_staggered` emits a `@warn` and skips that cohort
  entirely rather than erroring. If final pooled results look off,
  check the warnings for skipped cohorts before trusting the estimate.
- **Cross-cohort bootstrap pooling truncates to the shortest cohort's
  valid-draw count** (`B = minimum(length(cr.att_boot) for cr in
  cohort_results)`), after already dropping `NaN` draws (failed fits)
  within each cohort. With small `nboot` or unstable fits, `B` can end
  up much smaller than `nboot` — the effective bootstrap sample size is
  silently reduced, not reported anywhere in `EndidResult`.
- **`dvar` vs. auto-detection in `endid()`.** If you don't pass `dvar`,
  treatment status is inferred as "any unit with a `post == 1` row" —
  this means every unit with a post-period observation is treated;
  there is no control group unless you either restrict `df` to
  treated+control units beforehand or pass an explicit `dvar` column.
  Easy to accidentally run a treated-only sample if you forget `dvar`.
- Random-effects note: `endid`/`endid_staggered` call `Random.seed!`
  globally if `seed` is given — this seeds the RNG used inside a
  `Threads.@threads` bootstrap loop, so bootstrap draws are not exactly
  reproducible across different thread counts.
