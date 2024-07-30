include("smc_abc.jl")


function ABC(data::Array{Float64}, setup::Dict, seed::Int)
    moments_emp = calculate_moment_vector(data, setup) # calculate empirical moments 
    # calculate SMM weighting matrix
    weights = calculate_weights(setup, data)

    n_par = length(setup["mod"]["cali"]) # number of parameters
    n_particles = setup["opt"]["particles"] # number of particles
    simlen = length(data) * setup["opt"]["simfactor"]

    # initialize particles
    particles = zeros(Float64, n_par, n_particles) # estimated parameters
    particles = initialize_particles(particles, setup)
    particles = SharedArray(particles)

    particles_J = SharedArray{Float64}(n_particles) # J-values

    # estimate parameters for the i-th column of the matrix of observations
    particles, particles_J, adjusted_particles, adjusted_particles_J, total_sims = SMC_ABC(particles, particles_J, moments_emp, weights, setup, simlen, seed)
    
    return particles, particles_J, adjusted_particles, adjusted_particles_J, total_sims
end
