# Endid.jl: Distributional Difference-in-Differences

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`Endid.jl` provides a Julia implementation of **Distributional Difference-in-Differences (DiD)** using **Engression**. This method follows the approach proposed by **Lee & Wooldridge (2025)**, leveraging the power of stochastic neural networks to estimate entire treatment effect distributions.

Unlike standard DiD, which typically focuses on the Average Treatment Effect on the Treated (ATT), `Endid.jl` allows for estimating the **Quantile Treatment Effects (QTE)** and handling non-parallel trends through distributional modeling.

## Installation

This package requires `Engression.jl` as a dependency.

```julia
using Pkg
# Ensure Engression is available first
Pkg.add(url="https://github.com/xiangao/Engression.jl")
Pkg.add(url="https://github.com/xiangao/Endid.jl")
```

## Quickstart

### Common-Timing DiD

For panel data where treatment begins at the same time for all treated units. By default, `endid` identifies treated units as those having any `post == 1` observation, but you can also provide a specific treatment indicator column via `dvar`.

```julia
using Endid
using DataFrames
using Plots

# Assuming you have a panel DataFrame `df`
# y: outcome, id: unit id, time: time period, post: treatment indicator (0 pre, 1 post)
# optional: dvar: binary indicator for being in the treated group

res = endid(df, :outcome, :id, :time, :post; 
            dvar=:treated_group, # optional
            controls=[:age, :income], 
            nboot=100)

# Print results (ATT and QTE)
println(res)

# Visualize Quantile Treatment Effects with 95% CI
plot(res)
```

### Staggered Adoption DiD

For panel data where units receive treatment at different times.

```julia
using Endid
using DataFrames

# gvar: column indicating the first period of treatment (missing for never-treated)
# Example: if unit 1 is treated in period 5, gvar for unit 1 is 5.
res_staggered = endid_staggered(df, :outcome, :id, :time, :gvar;
                                controls=[:pre_trend], 
                                nboot=50)

# Results are pooled across cohorts using exposure-weighted averaging
println(res_staggered)
plot(res_staggered)
```

## Key Features

- **Lee & Wooldridge (2025) Framework:** Implements the latest advances in distributional DiD using panel data transformations.
- **Quantile Treatment Effects:** Provides full-distribution counterfactual comparisons.
- **Staggered Support:** Efficiently pools estimates across multiple treatment cohorts.
- **Parallel Bootstrap:** Uses Julia's multi-threading for fast inference.
- **Plots Integration:** Easy visualization of QTE with 95% confidence intervals.

## Vignettes

Full documentation: **https://xiangao.github.io/Endid.jl/**

| Vignette | Description |
|----------|-------------|
| [Comparison with Linear DiD](https://xiangao.github.io/Endid.jl/vignettes/01_comparison/) | Synthetic example comparing distributional and linear DiD targets |
| [Castle Doctrine Example](https://xiangao.github.io/Endid.jl/vignettes/02_castle_doctrine/) | Replication-style workflow using staggered treatment timing |

## Core API

### `endid(df, y, id, time, post; kwargs...)`
- `y`: Outcome column name.
- `id`: Unit identifier column.
- `time`: Time period column.
- `post`: Binary indicator for treatment (0 for pre, 1 for post).
- `controls`: List of control variable names (averaged over the pre-treatment period).
- `nboot`: Number of bootstrap iterations for inference.
- `rolling`: Panel transformation method ("demean" or "detrend").

### `endid_staggered(df, y, id, time, gvar; kwargs...)`
- `gvar`: Column containing the first treatment period for each unit.

### `EndidResult`
The result object returned by `endid` and `endid_staggered`:
- `.att`: Average Treatment Effect on the Treated.
- `.se`: Standard error of ATT.
- `.ci`: 95% Confidence Interval for ATT.
- `.qte`: DataFrame with Quantile Treatment Effects.
- `.model`: The underlying `Engressor` model.

## License
MIT
