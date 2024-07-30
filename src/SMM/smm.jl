@everywhere include("smm_optimization.jl")



function smm(data::Array{Float64}, setup::Dict, seed::Int)
    n_par = n_par = length(setup["mod"]["cali"]) # number of parameters
    n_rep = setup["opt"]["rep"] # number of repetitions

    # initialize shared arrays
    results_par = SharedArray{Float64}(n_par, n_rep) # estimated parameters
    results_j = SharedArray{Float64}(n_rep) # J-values
    results_iter = SharedArray{Float64}(n_rep) # number of iterations

    # calculate SMM weighting matrix
    weights = calculate_weights(setup, data)

    # distribute work among workers
    @sync @distributed for i in 1:n_rep
        Random.seed!(10000*i+seed) # set random number generator seed

        # estimate parameters for the i-th column of the matrix of observations
        results_par[:,i], results_j[i], results_iter[i] = smm_estimation(data, setup, weights)
        GC.gc()
    end

    return (Array(results_par), Array(results_j), Array(results_iter))
end
