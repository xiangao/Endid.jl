using Endid
using Test
using DataFrames
using Statistics
using Random

@testset "Endid.jl" begin
    Random.seed!(42)
    
    # 1. Test Panel Transformation
    n_units = 10
    n_time = 5
    df = DataFrame(
        id = repeat(1:n_units, inner=n_time),
        time = repeat(1:n_time, outer=n_units),
        post = repeat([0, 0, 1, 1, 1], outer=n_units)
    )
    # y = id_effect + time_effect + noise
    df.y = Float32.(df.id .+ df.time .+ randn(nrow(df)) .* 0.1)
    
    tdf = Endid.apply_transform(df, :y, :id, :time, :post, "demean")
    @test "ydot" in names(tdf)
    @test "ydot_postavg" in names(tdf)
    @test sum(tdf.firstpost) == n_units
    
    # 2. Test Main Endid Function
    # Construct a simple treatment effect
    # D = 1 for units 1:5
    df.D = [u <= 5 ? 1 : 0 for u in df.id]
    df.y .+= df.D .* df.post .* 2.0f0 # ATT = 2.0
    
    # Use very few epochs and nboot for fast testing
    result = endid(df, :y, :id, :time, :post; 
                   dvar=:D, nboot=5, num_epochs=100, hidden_dim=20)
    
    @test result.design == "common_timing"
    @test result.att > 0
    @test size(result.qte, 1) == 9 # default quantiles 0.1:0.1:0.9
    @test result.se > 0
end
