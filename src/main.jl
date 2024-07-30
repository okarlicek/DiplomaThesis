using Random
using Statistics
using LinearAlgebra
@everywhere using Distributions
@everywhere using BlackBoxOptim
@everywhere using StatsBase
@everywhere using SharedArrays
@everywhere include("utils/fcn_data.jl")
@everywhere include("utils/fcn_moments.jl")
@everywhere include("utils/fcn_moment_loss.jl")
@everywhere include("utils/fcn_loglikelihood.jl")
@everywhere include("utils/fcn_noise.jl")
include("utils/fcn_weights.jl")
include("utils/fcn_init_particles.jl")
include("utils/fcn_result.jl")
include("ABC/abc.jl")
include("Bayes/bayes.jl")
include("NPMSLE/npmsle.jl")
include("SMM/smm.jl")


function main(setup::Dict, seed::Int, path::Union{Nothing, String}=nothing)
    Random.seed!(seed) # set random seed

    returns = false
    if setup["opt"]["type"] == "ABC" || setup["opt"]["type"] == "SMM"
        returns = true
    end

    data = generate_series(setup, setup["mod"]["obs"], setup["mod"]["burn"], setup["mod"]["cali"], returns)
    
    # Running optimization algorithm
    if setup["opt"]["type"] == "ABC"
        results = ABC(data, setup, seed)
    elseif setup["opt"]["type"] == "Bayes"
        results = Bayes(data, setup, seed)
    elseif setup["opt"]["type"] == "NPMSLE"
        results = npmsle(data, setup, seed)
    elseif setup["opt"]["type"] == "SMM"
        results = smm(data, setup, seed)
    else
        throw("Unimplemented optimization method.")
    end

    # saving results if the path is specified
    if !isnothing(path)
        save_results(setup, results, path)
    end

    return results
end

