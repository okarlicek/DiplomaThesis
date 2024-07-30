include("smc_bayes.jl")


function Bayes(data::Array{Float64}, setup::Dict, seed::Int)
    n_par = length(setup["mod"]["cali"]) # number of parameters
    n_particles = setup["opt"]["particles"] # number of particles

    # initialize arrays
    particles = zeros(Float64, n_par, n_particles) # estimated parameters
    particles = initialize_particles(particles, setup)
    particles = SharedArray(particles)

    weights = ones(n_particles)
    normed_weights = zeros(n_particles)
    log_likelihoods = SharedArray{Float64}(n_particles)

    # estimate parameters for the i-th column of the matrix of observations
    particles, log_likelihoods, weights, total_sims = SMC_Bayes(data, particles, log_likelihoods, weights, normed_weights, setup, seed)
    
    return Array(particles), Array(log_likelihoods), weights, total_sims
end
