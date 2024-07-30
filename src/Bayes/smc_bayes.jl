using LogExpFunctions


function SMC_Bayes(data::Array{Float64, 1}, particles::SharedArray{Float64, 2}, log_likelihoods::SharedArray{Float64, 1}, 
                   weights::Array{Float64, 1}, normed_weights::Array{Float64, 1}, setup::Dict, seed::Int)
    # fixing random noise
    noise = generate_random_noise(setup)
    
    n_particles = setup["opt"]["particles"]
    c = 1.
    acc_rate = 1.

    # calculating loss values
    @sync @distributed for i in axes(particles)[2]
        log_likelihoods[i] = calculate_loglikelihood(particles[:, i], data, noise, setup)
    end
    total_sims = n_particles
    
    # Recursion
    for rec_i in 2:setup["opt"]["iter"]
        # SMC iteration
        c, acc_rate = recursion(data, setup, noise, particles, log_likelihoods, weights, normed_weights, rec_i, c, acc_rate, seed)
        total_sims += n_particles
    end

    return particles, log_likelihoods, weights, total_sims
end


function recursion(data::Array{Float64, 1}, setup::Dict, noise::Array{Float64}, particles::SharedArray{Float64, 2}, log_likelihoods::SharedArray{Float64, 1}, 
                   weights::Array{Float64, 1}, normed_weights::Array{Float64, 1}, rec_i::Int, c::Float64, acc_rate::Float64, seed::Int)
    # Correction
    phi_s = correction(log_likelihoods, weights, normed_weights, rec_i, setup["opt"]["iter"], setup["opt"]["particles"])
    
    cov_mat = weighted_covariance_matrix(particles, normed_weights) # cov matrix is based on particles from past iteration with new weights
    # Selection
    selection_bayes(particles, log_likelihoods, weights, normed_weights, setup["opt"]["particles"], rec_i == setup["opt"]["iter"])
    # Mutation
    c = calculate_c(c, acc_rate)  # covariance matrix scaler
    gaussian_inovation = MvNormal(cov_mat .* (c^2))

    acc_rate = mutation(data, setup, noise, particles, log_likelihoods, gaussian_inovation, phi_s, rec_i, seed)
    return c, acc_rate
end


function correction(log_likelihoods::SharedArray{Float64, 1}, weights::Array{Float64, 1}, normed_weights::Array{Float64, 1}, 
                    rec_i::Int, n_recursions::Int, n_particles::Int)
    # curent and last phi
    phi_s = calculate_phi(rec_i, n_recursions)
    phi_s_1 = calculate_phi(rec_i - 1, n_recursions)
    # incremental log-weights
    log_inc_weights = log_likelihoods .* (phi_s - phi_s_1)
    # normed weights
    normed_weights .= exp.(log_inc_weights + log.(weights) .- log(1/n_particles) .- logsumexp(log_inc_weights + log.(weights)))

    return phi_s
end


function selection_bayes(particles::SharedArray{Float64, 2}, log_likelihoods::SharedArray{Float64, 1}, weights::Array{Float64, 1}, 
                   normed_weights::Array{Float64, 1}, n_particles::Int, last_iter::Bool)
    # Effective sample size
    ESS = n_particles / (sum(normed_weights .^ 2) / n_particles)

    if ESS >= n_particles / 2 && !last_iter
        # keeping last iteration
        weights .= normed_weights
    else
        # systematic resampling
        systematic_resampling(particles, normed_weights, weights, log_likelihoods)
    end
end


function mutation(data::Array{Float64, 1}, setup::Dict, noise::Array{Float64}, particles::SharedArray{Float64, 2}, 
                  log_likelihoods::SharedArray{Float64, 1}, gaussian_inovation::Distribution, phi_s::Float64, rec_i::Int, seed::Int)
    # counter variable
    acc = SharedArray{Int}(1)

    @sync @distributed for i in 1:setup["opt"]["particles"]
        Random.seed!(10000 * i + seed + rec_i + 1) # set random number generator seed
        # proposing new particle
        proposed_particle = particles[:, i]
        proposed_particle = proposed_particle + rand(gaussian_inovation)
        # log-likelihood of proposed particle    
        proposed_ll = calculate_loglikelihood(proposed_particle, data, noise, setup)
        # log probability of accepting the new particle
        log_prob = phi_s * (proposed_ll - log_likelihoods[i]) + calc_log_prob(proposed_particle, setup) - calc_log_prob(particles[:, i], setup)        
        log_prob = min(0., log_prob)

        if log(rand()) < log_prob
            # accepting proposed particle
            particles[:, i] = proposed_particle
            log_likelihoods[i] = proposed_ll
            acc[1] += 1            
        end
    end

    acc_rate = acc[1] / setup["opt"]["particles"]
    return acc_rate
end


@everywhere function calc_log_prob(particle::Array{Float64, 1}, setup::Dict)
    log_prior = 0.
    # assuming uniform prior
    for i in axes(setup["mod"]["cons"])[1]
        lower, upper = setup["mod"]["cons"][i]
        if lower <= particle[i] <= upper
            log_prior += log(1 / (upper - lower))
        else
            return -Inf
        end
    end
    return log_prior
end

function calculate_phi(s::Int, s_total::Int, lambda::Float64=3.4)
    phi = (s - 1) / (s_total - 1)
    return phi ^ lambda
end


function systematic_resampling(particles::SharedArray{Float64, 2}, normed_weights::Array{Float64, 1}, 
                               weights::Array{Float64, 1}, log_likelihoods::SharedArray{Float64, 1})
    # setting arrays for resampled particles
    n = length(weights)
    new_particles = SharedArray{Float64}(size(particles))
    new_weights = zeros(n)
    new_log_likelihoods = SharedArray{Float64}(n)

    # drawing random number for systematic resampling
    r = rand()
    index = 1
    cum_weight = normed_weights[1]
    for i in 1:n
        # going through intervals one by one
        while r > cum_weight
            index += 1
            cum_weight += normed_weights[index]
        end
        
        new_particles[:, i] = particles[:, index]
        new_weights[i] = 1.
        new_log_likelihoods[i] = log_likelihoods[index]

        # Move to the next interval
        r += 1.0
    end
    particles .= new_particles
    weights .= new_weights
    log_likelihoods .= new_log_likelihoods
end

function weighted_covariance_matrix(X::SharedArray{Float64, 2}, weights::Array{Float64, 1})
    n = size(X, 2)  # number of observations

    mean_X = X * weights / sum(weights) # mean vector

    # Center the data
    centered_X = X .- mean_X

    # Compute the weighted covariance matrix
    weighted_cov_matrix = zeros(eltype(X), size(X, 1), size(X, 1))
    total_weight = sum(weights)

    for i in 1:n
        weighted_cov_matrix .+= weights[i] / total_weight * centered_X[:, i] * centered_X[:, i]'
    end
    # there can be rounding precision error, therefore making sure opposite values are the same
    for i in 1:size(X, 1)
        for j in 1:size(X, 1)
            weighted_cov_matrix[i, j] = weighted_cov_matrix[j, i]
        end
    end

    return weighted_cov_matrix
end


function calculate_c(last_c::Float64, acc_rate::Float64)
    x = acc_rate
    f = 0.95 + 0.1 * (exp(16 * (x-0.25)) / (1 + exp(16 * (x-0.25))))
    c = last_c * f
    return c
end
