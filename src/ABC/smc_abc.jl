include("particle_adjustment.jl")


function SMC_ABC(particles::SharedArray{Float64, 2}, particles_J::SharedArray{Float64, 1}, moments_emp::Array{Float64, 1}, weights::Array{Float64, 2}, 
                setup::Dict, simlen::Int, seed::Int, c::Float64=0.01, verbose::Bool=false)
    total_sims = 0
    maximum_iterations = setup["opt"]["total_iter"]
    # selection breakpoint
    breakpoint = trunc(Int, setup["opt"]["particles"] * setup["opt"]["tau"])

    # calculating particles losses
    @sync @distributed for i in axes(particles)[2]
        Random.seed!(10000*i+seed)
        particles_J[i] = moment_loss_function(particles[:, i], setup, moments_emp, weights, simlen)
    end
    total_sims += setup["opt"]["particles"]

    delta_max = maximum(particles_J)  # current max loss
    R = calculate_R(setup["opt"]["particles"], setup["opt"]["particles"], c)

    iteration = 0
    # Optimization loop
    while total_sims < maximum_iterations
        iteration += 1
        # SMC iteration
        particles, particles_J, delta_max, delta, R, iter_total_tries, iter_successful_tries = SMC_step(particles, particles_J, R, c, breakpoint, setup, 
                                                                                                        moments_emp, weights, simlen, iteration, seed)

        total_sims += iter_total_tries
        if verbose
            print_SMC_step_progress(iteration, delta_max, delta, R, iter_total_tries, iter_successful_tries, total_sims)
        end
    end

    # Linear adjustment of particles
    adjusted_particles, adjusted_particles_J = nothing, nothing
    try
        # particle adjustment
        adjusted_particles = SharedArray(adjust_particles(particles, particles_J, moments_emp, simlen, delta_max))
        adjusted_particles_J = SharedArray{Float64}(size(particles_J))
        # loss calculation
        @sync @distributed for i in eachindex(particles_J)
            Random.seed!(10000 * i + seed + iteration + 1) # set random number generator seed
            adjusted_particles_J[i] = moment_loss_function(adjusted_particles[:, i], setup, moments_emp, weights, simlen)
        end
    catch
        adjusted_particles = particles
        adjusted_particles_J = particles_J
    end
    
    return Array(particles), Array(particles_J), Array(adjusted_particles), Array(adjusted_particles_J), total_sims
end


function SMC_step(particles::SharedArray{Float64, 2}, particles_J::SharedArray{Float64, 1}, R::Float64, c::Float64, breakpoint::Int, 
                  setup::Dict, moments_emp::Array{Float64, 1}, weights::Array{Float64, 2}, simlen::Int, iteration::Int, seed::Int)
    # keeping count of iterations
    iter_total_tries, iter_successful_tries = SharedArray{Int}(1), SharedArray{Int}(1)
    
    particles, particles_J, delta = selection_abc(particles, particles_J, breakpoint) # Selection of particles
    MCMCkernel = GaussianInnovation(particles[:, 1:breakpoint])  # Normal distribution based on selected particles 

    @sync @distributed for j in breakpoint + 1: size(particles)[2]
        # updating particles behind breakpoint
        Random.seed!(10000 * j + seed + iteration) # set random number generator seed

        idx = rand(1:breakpoint)
        particles[:, j] = particles[:, idx]
        # MCMC move update
        particle, particle_J, total_tries, successful_tries = update_particle(particles[:, idx], particles_J[idx], R, delta, 
                                                                              MCMCkernel, setup, moments_emp, weights, simlen)
        
        # saving updated particle
        particles[:, j] = particle
        particles_J[j] = particle_J
        iter_total_tries[1] += total_tries
        iter_successful_tries[1] += successful_tries
    end
    
    delta_max = maximum(particles_J) # new maximal loss
    R = calculate_R(iter_successful_tries[1], iter_total_tries[1], c) # new R

    return particles, particles_J, delta_max, delta, R, iter_total_tries[1], iter_successful_tries[1]
end


function selection_abc(particles::SharedArray{Float64, 2}, particles_J::SharedArray{Float64, 1}, breakpoint::Int)
    # sorting particles
    sorted_idxs = sortperm(particles_J)
    # the indexing changes SharedArray into Array, therefore we need to transform it back
    particles_J = SharedArray(particles_J[sorted_idxs])
    particles = SharedArray(particles[:, sorted_idxs])
    # breakpoint delta
    delta = particles_J[breakpoint]
    return particles, particles_J, delta
end


@everywhere function update_particle(particle::Array{Float64, 1}, particle_J::Float64, R::Float64, delta::Float64, MCMCkernel::Distribution, 
                                    setup::Dict, moments_emp::Array{Float64, 1}, weights::Array{Float64, 2}, simlen::Int)    
    total_tries = 0
    successful_tries = 0

    for _ in 1:R
        # proposing new particle based of MCMC move
        proposed_particle = particle + rand(MCMCkernel)
        
        zero_prior = false
        for i in axes(setup["mod"]["cons"])[1]
            # checking if proposed particle is inside prior range
            if proposed_particle[i] < setup["mod"]["cons"][i][1] || proposed_particle[i] > setup["mod"]["cons"][i][2]
                zero_prior = true
            end
        end

        if zero_prior
            # skipping particles outside uniform prior
            continue
        end

        proposed_J = moment_loss_function(proposed_particle, setup, moments_emp, weights, simlen) # loss of new particle
        
        total_tries += 1
        if proposed_J <= delta
            # updating particle if the loss is better than breakpoint delta
            successful_tries += 1
            particle_J = proposed_J
            particle = proposed_particle
        end
    end

    return particle, particle_J, total_tries, successful_tries
end


function calculate_R(successful_tries::Int, total_tries::Int, c::Float64)
    # number of update iterations to be sure that every particle is updated with some probability
    if successful_tries == 0
        successful_tries = 1
    end

    R = round(log(c) / log(1 - (successful_tries / (total_tries+1))))
    return max(R, 1)
end


function GaussianInnovation(selected_particles::Array{Float64, 2})
    # Normal distribution for MCMC kernel move
    cov_mat = cov(selected_particles, dims=2) .* 2
    d = MvNormal(cov_mat)
    return d
end

function print_SMC_step_progress(iteration::Int, delta_max::Float64, delta::Float64, R::Float64, iter_total_tries::Int, iter_successful_tries::Int, total_sims::Int)
    # progress printer
    println("Iteration $iteration")
    println("Current delta $delta")
    println("N tries = $iter_total_tries ; successful tries = $iter_successful_tries ; total_sims = $total_sims")
    println("Current max delta $delta_max")
    println("New R = $R")
    println("___________________________________________________________________")
end
