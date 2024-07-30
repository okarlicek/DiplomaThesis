using JLD

function save_results(setup::Dict, results::Tuple, path::String)
    mkpath(dirname(path))

    if setup["opt"]["type"] == "ABC"
        save_abc(results, path)
    elseif setup["opt"]["type"] == "Bayes"
        save_bayes(results, path)
    elseif setup["opt"]["type"] == "NPMSLE"
        save_npmsle(results, path)
    elseif setup["opt"]["type"] == "SMM"
        save_smm(results, path)
    else
        throw("Unimplemented optimization method.")
    end
end


function save_abc(results::Tuple, path::String)
    particles, particles_J, adjusted_particles, adjusted_particles_J, total_sims = results

    # Open the JLD file in write mode
    jldopen(path, "w") do file
        # Save arrays into the file
        file["particles"] = particles
        file["particles_J"] = particles_J
        file["adjusted_particles"] = adjusted_particles
        file["adjusted_particles_J"] = adjusted_particles_J
        file["total_sims"] = total_sims
    end
end


function save_bayes(results::Tuple, path::String)
    particles, log_likelihoods, weights, total_sims = results

    # Open the JLD file in write mode
    jldopen(path, "w") do file
        # Save arrays into the file
        file["particles"] = particles
        file["particles_J"] = log_likelihoods
        file["weights"] = weights
        file["total_sims"] = total_sims
    end
end


function save_npmsle(results::Tuple, path::String)
    parameters, parameters_j, total_sims = results
    
    # Open the JLD file in write mode
    jldopen(path, "w") do file
        # Save arrays into the file
        file["parameters"] = parameters
        file["parameters_J"] = parameters_j
        file["total_sims"] = total_sims
    end
end


function save_smm(results::Tuple, path::String)
    parameters, parameters_j, total_sims = results

    # Open the JLD file in write mode
    jldopen(path, "w") do file
        # Save arrays into the file
        file["parameters"] = parameters
        file["parameters_J"] = parameters_j
        file["total_sims"] = total_sims
    end
end