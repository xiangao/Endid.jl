# Castle Doctrine Example

```@meta
CurrentModule = Endid
```

This vignette applies `Endid.jl` to the Castle Doctrine dataset with staggered adoption timing.

## Data Preparation

```@example endid_castle_data
using Endid
using DataFrames
using CSV
using Statistics
using Plots
using Random

Random.seed!(42)

castle = CSV.read(joinpath(@__DIR__, "..", "..", "..", "vignettes", "data", "castle.csv"), DataFrame)
castle.gvar = [y == 0 ? missing : y for y in castle.effyear]
controls = ["poverty", "unemployrt", "blackm_15_24", "whitem_15_24"]

cohorts = combine(groupby(castle, :sid), :gvar => first => :gvar)
combine(groupby(cohorts, :gvar), :sid => length => :count)
```

## Distributional Estimation

```@example endid_castle
fit_endid = endid_staggered(
    castle,
    :lhomicide,
    :sid,
    :year,
    :gvar;
    controls = controls,
    rolling = "demean",
    num_epochs = 100,
    nboot = 5,
    seed = 42,
)

println(fit_endid)
```

## Quantile Treatment Effects

```@example endid_castle
p = plot(fit_endid)
savefig(p, "castle_qte.svg")
nothing
```

![](castle_qte.svg)
