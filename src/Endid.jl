module Endid

using DataFrames
using Statistics
using ShiftedArrays
using Engression
using ProgressMeter
using Random
using Base.Threads
using RecipesBase

export endid, endid_staggered, EndidResult

struct EndidResult
    att::Float64
    se::Float64
    ci::Tuple{Float64, Float64}
    qte::DataFrame
    model::Engressor
    design::String
end

function Base.show(io::IO, res::EndidResult)
    println(io, "Endid Result ($(res.design) design)")
    println(io, "-" ^ 30)
    println(io, "ATT Estimate: ", round(res.att, digits=4))
    println(io, "Std. Error  : ", round(res.se, digits=4))
    println(io, "95% CI      : (", round(res.ci[1], digits=4), ", ", round(res.ci[2], digits=4), ")")
    println(io, "\nQuantile Treatment Effects (QTE):")
    show(io, res.qte, allrows=true)
end

@recipe function f(res::EndidResult)
    q = res.qte.quantile
    eff = res.qte.effect
    se = res.qte.se
    
    title --> "Quantile Treatment Effects (QTE)"
    xlabel --> "Quantile"
    ylabel --> "Effect"
    
    # Plot confidence ribbon
    @series begin
        seriestype := :line
        fillrange := eff .- 1.96 .* se
        fillalpha := 0.2
        label := "95% CI"
        color := :blue
        q, eff .+ 1.96 .* se
    end
    
    # Plot point estimates
    @series begin
        seriestype := :line
        linewidth := 2
        label := "QTE"
        color := :blue
        marker := :circle
        q, eff
    end
    
    # Plot horizontal line at 0
    @series begin
        seriestype := :hline
        linestyle := :dash
        color := :black
        label := ""
        [0]
    end
    
    # Plot ATT
    @series begin
        seriestype := :hline
        linestyle := :dot
        color := :red
        label := "ATT"
        [res.att]
    end
end

"""
    apply_transform(df, y, id, time, post, rolling; tpost1=nothing)

Apply Lee & Wooldridge (2025) panel transformations.
"""
function apply_transform(df, y, id, time, post, rolling; tpost1=nothing)
    # Ensure columns are sorted
    sdf = sort(df, [id, time])
    
    # Define unit-level transformation
    function unit_transform(sub_y, sub_t, sub_p)
        idx_pre = findall(p -> !ismissing(p) && p == 0, sub_p)
        
        if isempty(idx_pre)
            return fill(missing, length(sub_y))
        end
        
        if rolling == "demean"
            y_pre_mean = mean(skipmissing(sub_y[idx_pre]))
            return [ismissing(y) ? missing : y - y_pre_mean for y in sub_y]
        elseif rolling == "detrend"
            if length(idx_pre) < 2
                return fill(missing, length(sub_y))
            end
            t_pre = sub_t[idx_pre]
            y_pre = sub_y[idx_pre]
            
            # Simple linear regression y ~ t
            t_m = mean(t_pre)
            tc_pre = t_pre .- t_m
            beta = sum(tc_pre .* y_pre) / sum(tc_pre.^2)
            alpha = mean(y_pre) .- beta .* t_m
            
            y_hat = alpha .+ beta .* sub_t
            return [ismissing(y) ? missing : y - yh for (y, yh) in zip(sub_y, y_hat)]
        else
            error("Unknown rolling method: $rolling")
        end
    end
    
    # Apply transformation by group
    sdf = transform(groupby(sdf, id), [y, time, post] => ( (y, t, p) -> unit_transform(y, t, p) ) => :ydot)
    
    # Compute post-treatment average of ydot per unit
    post_df = filter(row -> !ismissing(row[post]) && row[post] == 1, sdf)
    unit_avgs = combine(groupby(post_df, id), :ydot => (x -> mean(skipmissing(x))) => :ydot_postavg)
    
    # Merge back
    sdf = leftjoin(sdf, unit_avgs, on=id)
    
    # Mark firstpost
    if isnothing(tpost1)
        tpost_rows = sdf[(.!ismissing.(sdf[!, post])) .& (sdf[!, post] .== 1), time]
        tpost1 = isempty(tpost_rows) ? nothing : minimum(tpost_rows)
    end
    
    if isnothing(tpost1)
        sdf[!, :firstpost] .= false
    else
        sdf[!, :firstpost] = [(row[time] == tpost1) && !ismissing(row.ydot_postavg) for row in eachrow(sdf)]
    end
    
    return sdf
end

"""
    fit_engression_cs(Y, D, X_ctrl; kwargs...)

Internal helper to fit engression on cross-section and return predictions.
Calculates ATT by averaging unit-level counterfactual differences.
Calculates QTE by comparing quantiles of pooled counterfactual samples.
"""
function fit_engression_cs(Y, D, X_ctrl;
                           quantiles=0.1:0.1:0.9, num_epochs=1000, lr=0.001,
                           noise_dim=10, hidden_dim=100, num_layers=3, nsample=500)
    X = D
    if !isnothing(X_ctrl)
        X = hcat(D, X_ctrl)
    end
    
    model = engression(X, Y; 
                       num_layers=num_layers, hidden_dim=hidden_dim, noise_dim=noise_dim,
                       num_epochs=num_epochs, lr=lr, standardize=true)
    
    # Indices of treated units in this cross-section
    idx_treated = findall(d -> d == 1.0, D)
    if isempty(idx_treated)
        return (att=NaN, qte=fill(NaN, length(quantiles)), model=model)
    end
    
    # Build counterfactual X matrices for treated units
    if isnothing(X_ctrl)
        X1 = fill(1.0f0, length(idx_treated), 1)
        X0 = fill(0.0f0, length(idx_treated), 1)
    else
        ctrl_treated = X_ctrl[idx_treated, :]
        X1 = hcat(fill(1.0f0, length(idx_treated)), ctrl_treated)
        X0 = hcat(fill(0.0f0, length(idx_treated)), ctrl_treated)
    end
    
    # ATT: average difference in conditional means
    yhat1 = predict(model, X1, target="mean", sample_size=nsample)
    yhat0 = predict(model, X0, target="mean", sample_size=nsample)
    att = mean(yhat1 .- yhat0)
    
    # QTE: quantiles of pooled counterfactual samples
    s1 = sample(model, X1, sample_size=nsample) # (out_dim, n_treated, nsample)
    s0 = sample(model, X0, sample_size=nsample)
    
    s1_pool = vec(s1)
    s0_pool = vec(s0)
    
    qte = [quantile(s1_pool, q) - quantile(s0_pool, q) for q in quantiles]
    
    return (att=att, qte=qte, model=model)
end

"""
    endid(df, y, id, time, post; kwargs...)

Main entry point for common-timing DiD.
"""
function endid(df, y, id, time, post;
               dvar=nothing, controls=nothing, rolling="demean", quantiles=0.1:0.1:0.9,
               nboot=100, nsample=500, num_epochs=1000, lr=0.001,
               noise_dim=10, hidden_dim=100, num_layers=3, seed=nothing)
    
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # 1. Transform panel
    tpost_rows = df[(.!ismissing.(df[!, post])) .& (df[!, post] .== 1), time]
    tpost1 = isempty(tpost_rows) ? nothing : minimum(tpost_rows)
    tdf = apply_transform(df, y, id, time, post, rolling; tpost1=tpost1)
    
    # 2. Extract cross-section
    cs = filter(row -> row.firstpost, tdf)
    
    # Treatment indicator
    if isnothing(dvar)
        treated_units = unique(df[(.!ismissing.(df[!, post])) .& (df[!, post] .== 1), id])
        cs[!, :D_] = [u in treated_units ? 1.0 : 0.0 for u in cs[!, id]]
    else
        cs[!, :D_] = Float64.(cs[!, dvar])
    end
    
    # Controls
    X_ctrl = nothing
    if !isnothing(controls)
        pre_df = filter(row -> !ismissing(row[post]) && row[post] == 0, tdf)
        ctrl_cs = combine(groupby(pre_df, id), [c => (x -> mean(skipmissing(x))) => Symbol(string(c, "_pre")) for c in controls])
        cs = leftjoin(cs, ctrl_cs, on=id)
        ctrl_names = [Symbol(string(c, "_pre")) for c in controls]
        cs = dropmissing(cs, ctrl_names)
        X_ctrl = Matrix{Float32}(cs[:, ctrl_names])
    end
    
    cs = dropmissing(cs, :ydot_postavg)
    Y_cs = Float32.(cs.ydot_postavg)
    D_cs = Float32.(cs.D_)
    
    # 3. Fit
    res = fit_engression_cs(Y_cs, D_cs, X_ctrl; 
                            quantiles=quantiles, num_epochs=num_epochs, lr=lr,
                            noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                            nsample=nsample)
    
    # 4. Bootstrap Inference
    att_boot = zeros(nboot)
    qte_boot = zeros(nboot, length(quantiles))
    
    println("Running Parallel Bootstrap (n=$nboot)...")
    p = Progress(nboot)
    
    Threads.@threads for b in 1:nboot
        idx = rand(1:nrow(cs), nrow(cs))
        Y_b = Y_cs[idx]
        D_b = D_cs[idx]
        X_ctrl_b = isnothing(X_ctrl) ? nothing : X_ctrl[idx, :]
        
        try
            res_b = fit_engression_cs(Y_b, D_b, X_ctrl_b; 
                                      quantiles=quantiles, num_epochs=num_epochs, lr=lr,
                                      noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                      nsample=nsample)
            att_boot[b] = res_b.att
            qte_boot[b, :] = res_b.qte
        catch
            att_boot[b] = NaN
        end
        next!(p)
    end
    
    valid = .!isnan.(att_boot)
    att_final = mean(att_boot[valid])
    se_overall = std(att_boot[valid])
    ci_overall = (quantile(att_boot[valid], 0.025), quantile(att_boot[valid], 0.975))
    
    qte_df = DataFrame(quantile=collect(quantiles))
    qte_df.effect = vec(mean(qte_boot[valid, :], dims=1))
    qte_df.se = vec(std(qte_boot[valid, :], dims=1))
    
    return EndidResult(att_final, se_overall, ci_overall, qte_df, res.model, "common_timing")
end

"""
    endid_staggered(df, y, id, time, gvar; kwargs...)

Staggered adoption DiD. `gvar` is the first treatment period (NA/missing for never treated).
"""
function endid_staggered(df, y, id, time, gvar;
                         controls=nothing, rolling="demean", quantiles=0.1:0.1:0.9,
                         nboot=100, num_epochs=1000, lr=0.001,
                         noise_dim=10, hidden_dim=100, num_layers=3, seed=nothing, nsample=500)
    
    if !isnothing(seed)
        Random.seed!(seed)
    end

    sdf = copy(df)
    gvals = sdf[!, gvar]
    never_treated_units = unique(sdf[ismissing.(gvals), id])
    cohorts = sort(unique(collect(skipmissing(gvals))))
    
    if isempty(cohorts)
        error("No treatment cohorts found in gvar.")
    end
    if isempty(never_treated_units)
        error("No never-treated units found.")
    end

    cohort_results = []
    
    for g in cohorts
        treated_units = unique(sdf[(.!ismissing.(gvals)) .& (gvals .== g), id])
        keep_units = vcat(treated_units, never_treated_units)
        df_g = sdf[findall(u -> u in keep_units, sdf[!, id]), :]
        
        df_g[!, :post_cal] = [!ismissing(t) && t >= g ? 1 : 0 for t in df_g[!, time]]
        df_trans = apply_transform(df_g, y, id, time, :post_cal, rolling; tpost1=g)
        df_trans[!, :d_] = [u in treated_units ? 1.0 : 0.0 for u in df_trans[!, id]]
        
        cs = filter(row -> row.firstpost, df_trans)
        
        X_ctrl_cs = nothing
        if !isnothing(controls)
            pre_df = filter(row -> !ismissing(row.post_cal) && row.post_cal == 0, df_g)
            ctrl_cs = combine(groupby(pre_df, id), [c => (x -> mean(skipmissing(x))) => Symbol(string(c, "_pre")) for c in controls])
            cs = leftjoin(cs, ctrl_cs, on=id)
            ctrl_names = [Symbol(string(c, "_pre")) for c in controls]
            cs = dropmissing(cs, ctrl_names)
            X_ctrl_cs = Matrix{Float32}(cs[:, ctrl_names])
        end
        
        cs = dropmissing(cs, :ydot_postavg)
        
        if nrow(cs) < 4 || sum(cs.d_ .== 1) < 2 || sum(cs.d_ .== 0) < 2
            @warn "Cohort $g: skipping (degenerate cross-section)."
            continue
        end
        
        Y_cs = Float32.(cs.ydot_postavg)
        D_cs = Float32.(cs.d_)
        
        # Fit Point Estimate
        res_g = fit_engression_cs(Y_cs, D_cs, X_ctrl_cs; 
                                  quantiles=quantiles, num_epochs=num_epochs, lr=lr,
                                  noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                  nsample=nsample)
        
        # Bootstrap
        att_boot = zeros(nboot)
        qte_boot = zeros(nboot, length(quantiles))
        
        for b in 1:nboot
            idx = rand(1:nrow(cs), nrow(cs))
            Y_b = Y_cs[idx]
            D_b = D_cs[idx]
            X_ctrl_b = isnothing(X_ctrl_cs) ? nothing : X_ctrl_cs[idx, :]
            
            try
                res_b = fit_engression_cs(Y_b, D_b, X_ctrl_b; 
                                          quantiles=quantiles, num_epochs=num_epochs, lr=lr,
                                          noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                          nsample=nsample)
                att_boot[b] = res_b.att
                qte_boot[b, :] = res_b.qte
            catch
                att_boot[b] = NaN
            end
        end
        
        valid = .!isnan.(att_boot)
        if any(valid)
            push!(cohort_results, (
                cohort = g,
                n_treated = length(treated_units),
                att_boot = att_boot[valid],
                qte_boot = qte_boot[valid, :],
                model = res_g.model
            ))
        end
    end
    
    if isempty(cohort_results)
        error("No cohorts could be estimated.")
    end
    
    n_treated_total = sum([cr.n_treated for cr in cohort_results])
    weights = [cr.n_treated / n_treated_total for cr in cohort_results]
    
    B = minimum([length(cr.att_boot) for cr in cohort_results])
    att_boot_pooled = zeros(B)
    qte_boot_pooled = zeros(B, length(quantiles))
    
    for i in 1:length(cohort_results)
        att_boot_pooled .+= weights[i] .* cohort_results[i].att_boot[1:B]
        qte_boot_pooled .+= weights[i] .* cohort_results[i].qte_boot[1:B, :]
    end
    
    att_final = mean(att_boot_pooled)
    se_overall = std(att_boot_pooled)
    ci_overall = (quantile(att_boot_pooled, 0.025), quantile(att_boot_pooled, 0.975))
    
    qte_df = DataFrame(quantile=collect(quantiles))
    qte_df.effect = vec(mean(qte_boot_pooled, dims=1))
    qte_df.se = vec(std(qte_boot_pooled, dims=1))
    
    return EndidResult(att_final, se_overall, ci_overall, qte_df, cohort_results[1].model, "staggered")
end

end # module
