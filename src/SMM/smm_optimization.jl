

function smm_estimation(data::Array{Float64,1}, setup::Dict, weights::Array)
    iter = setup["opt"]["iter"] # number of iterations
    search_range = setup["mod"]["cons"] # search constraints

    # data structures
    results_parm = zeros(length(search_range), setup["opt"]["inits"]) # matrix of estimated parameters
    results_j = zeros(setup["opt"]["inits"]) # array of J values
    results_iter = zeros(setup["opt"]["inits"]) # array to count number of calls of the objective function

    moments_emp = calculate_moment_vector(data, setup) # calculate empirical moments 

    for i in 1:setup["opt"]["inits"]
        # select parameters such that J value is minimized
        optout = bboptimize(theta -> moment_loss_function(theta, setup, moments_emp, weights, 
                                                          length(data)*setup["opt"]["simfactor"]),
                            SearchRange = search_range,
                            Method = :adaptive_de_rand_1_bin_radiuslimited,
                            NumDimensions = length(search_range),
                            MaxFuncEvals = iter,
                            TraceMode = :silent)

        results_parm[:,i] = best_candidate(optout) # retrieve found parameters
        results_j[i] = best_fitness(optout) # retrieve corresponding J value
        results_iter[i] = optout.f_calls
    end

    return (results_parm[:, argmin(results_j)], minimum(results_j), results_iter[argmin(results_j)])
end
