
function EpanechnikovKernel(x::Float64, h::Float64)
    x = x / h

    if abs(x) > 1
        return 0.0
    end

    kernel_value = (3/4) * (1 - x^2)
    return kernel_value / h
end

function adjust_particles(particles::SharedArray{Float64, 2}, particles_J::SharedArray{Float64, 1}, 
                          moments_emp::Array{Float64, 1}, simlen::Int, delta::Float64)
    # Linear regression adjustment
    moments_particles = SharedArray{Float64}((length(moments_emp), length(particles_J)))
    particles_weights = zeros(length(particles_J))
    adjusted_particles = zeros(size(particles))

    moments_sim = zeros(length(moments_emp), setup["opt"]["sim"]) # array of simulated moments
    @sync @distributed for i in axes(particles)[2]
        for j in 1:setup["opt"]["sim"]
            # simulate time series and calculate simulated moments
            sim_data = generate_series(setup, simlen, setup["mod"]["burn"], particles[:, i], true)
            moments_sim[:, j] = calculate_moment_vector(sim_data, setup)
        end
        moments_particles[:, i] = mean(moments_sim, dims=2)
    end

    for i in axes(particles_J)[1]
        # calculating weight of particles
        particles_weights[i] = EpanechnikovKernel(particles_J[i], delta)
    end

    # intercept + difference between moments
    x = vcat(ones(1, length(particles_J)), moments_particles .- moments_emp)
    # lin regression for each parameter
    for i in axes(particles)[1]
        # betas from regression
        betas = inv(x * diagm(particles_weights) * transpose(x))
        betas = betas * x * diagm(particles_weights) * particles[i, :]
        # adjustment
        adjusted_particles[i, :] = betas[1] .+ (particles[i, :] - transpose(x) * betas)
    end

    return adjusted_particles
end

