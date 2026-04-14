ENV["GKSwstype"] = "100"

using Documenter
using Endid

makedocs(
    sitename = "Endid.jl",
    modules = [Endid],
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Comparison" => "vignettes/01_comparison.md",
            "Castle Doctrine" => "vignettes/02_castle_doctrine.md",
        ],
    ],
    warnonly = true,
)
