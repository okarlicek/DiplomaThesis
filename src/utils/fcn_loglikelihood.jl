
function calculate_loglikelihood(theta::Array{Float64,1}, data::Array{Float64,1}, noise::Array{Float64}, setup::Dict)
    T = length(data)
    L = 0.
    # generating starting latent values
    latent = generate_latent_vector(setup, theta)
    past_noise = zeros(1) # for arma garch

    simulated_values = zeros(Float64, (setup["opt"]["traj"], setup["opt"]["sim"]))

    for i in 1:(T - setup["opt"]["traj"] + 1)
        # log-likelihood calculation
        if generate_model_step(setup, i, data, simulated_values, noise, theta, latent, past_noise)        
            L += kernel_density_estimation(data[i: i+setup["opt"]["traj"]-1], simulated_values)
        end
    end

    if L == - Inf || L == Inf || isnan(L)
        return nextfloat(typemin(Float64))
    end
    return L
end


function kernel_density_estimation(real_values::Array{Float64,1}, simulated_values::Array{Float64,2})
    n_sims = size(simulated_values)[2] # number of simulations
    dimension = size(simulated_values)[1] # trajectory length

    # inverse and determinant of upper (numerator) and lower (denominator) bandwidth matrix
    # we are working with diagonal bandwidth matrix
    # the upper matrix is y_t + conditioning set, lower matrix is conditioning set
    # however, we mostly use trajectory = 0 -> no conditioning set
    upper_b_inverse = zeros(dimension, dimension)
    upper_b_determinant = 1.
    lower_b_inverse = zeros(dimension - 1, dimension - 1)
    lower_b_determinant = 1.
    for i in 1:dimension
        sd = std(simulated_values[i, :]) # standard deviation of simulated values
        upper_value = ((4 / ((dimension + 2) * n_sims)) ^ (1 / (dimension + 4)) * sd) ^ 2
        upper_b_inverse[i, i] = 1 / upper_value  # inverse of the diagonal value
        upper_b_determinant *= upper_value  # updating determinant
        if i != dimension
            lower_value = ((4 / ((dimension + 1) * n_sims)) ^ (1 / (dimension + 3)) * sd) ^ 2
            lower_b_inverse[i, i] = 1 / lower_value  # inverse of the diagonal value
            lower_b_determinant *= lower_value  # updating determinant
        end
    end
    
    upper_sum = 0.
    lower_sum = 0.
    for i in 1:n_sims
        # numerator part
        upper_sum += multivariate_normal_kernel(real_values[1:dimension], simulated_values[1:dimension, i], upper_b_inverse, upper_b_determinant, dimension)
        if dimension > 1
            # denominator part
            lower_sum += multivariate_normal_kernel(real_values[1:dimension - 1], simulated_values[1:dimension - 1, i], lower_b_inverse, lower_b_determinant, dimension-1)
        else
            # when trajectory is equal to 1 (no conditioning set), then the KDE is mean of the numerator part
            lower_sum += 1.
        end
    end
    ll = log(upper_sum / lower_sum)
    if ll == - Inf || isnan(ll)
        return nextfloat(typemin(Float64))
    end
    return ll
end

function multivariate_normal_kernel(real_value::Array{Float64,1}, simulated::Array{Float64,1}, b_inverse::Array{Float64,2}, 
                                    b_determinant::Float64, dimension::Int)
    diff = real_value - simulated
    # multivariate normal kernel
    kernel_value = exp(-(transpose(diff) * b_inverse * diff) / 2)
    kernel_value *= (2 * pi)^(-dimension/2) * b_determinant^(-1/2)
    return kernel_value
end
