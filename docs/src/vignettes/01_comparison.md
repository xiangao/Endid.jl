# Comparing Endid and Linear DiD

```@meta
CurrentModule = Endid
```

This example illustrates a location-scale treatment effect, where the policy changes both the mean and dispersion of the outcome.

## Setup

```@example endid_comparison
using Endid
using DataFrames
using Statistics
using Random
using Distributions
using Plots

Random.seed!(42)

N = 80
T_total = 6
tpost1 = 4

unit = repeat(1:N, inner=T_total)
time = repeat(1:T_total, outer=N)
alpha = repeat(randn(N) .* 0.3, inner=T_total)
group = repeat(vcat(fill(1, Int(N / 2)), fill(0, Int(N / 2))), inner=T_total)
post = [t >= tpost1 ? 1 : 0 for t in time]
D = post .* group

epsilon = randn(N * T_total)
y = alpha .+ 0.5 .* D .+ (0.5 .+ 1.0 .* D) .* epsilon

df = DataFrame(unit = unit, time = time, y = y, post_treat = post, group = group)
first(df, 10)
```

## Estimation

```@example endid_comparison
fit_endid = endid(df, :y, :unit, :time, :post_treat;
                  dvar=:group, num_epochs=10, nboot=1, seed=42)

println(fit_endid)
```

## Quantile Treatment Effects

```@example endid_comparison
p = plot(fit_endid)
savefig(p, "endid_qte.svg")
nothing
```

![](endid_qte.svg)

Overlay the analytical QTE, `0.5 + Φ^{-1}(τ)`:

```@example endid_comparison
taus = 0.05:0.05:0.95
true_qte = 0.5 .+ quantile.(Normal(0, 1), taus)

p2 = plot(fit_endid)
plot!(p2, taus, true_qte, label="True QTE", linestyle=:dash, color=:red, linewidth=2)
savefig(p2, "endid_qte_true.svg")
nothing
```

![](endid_qte_true.svg)
