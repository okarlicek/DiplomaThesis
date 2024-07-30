# ARMA(1, 1)-GARCH(1, 1)


function generateARMAGARCH(n_observ::Int, burn_in::Int, theta::Array{Float64, 1}, fixed_params::Array{Float64, 1})
    pseudo_data = zeros(burn_in + n_observ)
    sigma_sq = 0.
    noise = 0.
    for i in 2:(n_observ + burn_in)
        p, sigma_sq, noise = ARMAGARCHstep(theta, fixed_params, pseudo_data[i-1], noise, sigma_sq)
        pseudo_data[i] = p
    end
    return pseudo_data[burn_in + 1: end]
end


function ARMAGARCHstep(theta::Array{Float64,1},  fixed_params::Array{Float64,1}, pt_1::Float64, noiset_1::Float64, sigmasqt_1::Float64, epsilon_t::Union{Nothing, Float64}=nothing)
    nu = theta[1] 
    a = theta[2]
    b = theta[3]
    alpha = theta[4]
    beta = theta[5]
    omega = theta[6]

    if epsilon_t === nothing
        epsilon_t = randn()
    end
    
    sigma_sq = omega + alpha * (noiset_1 ^ 2) + beta * sigmasqt_1
    if sigma_sq < 0
        return Inf, Inf, epsilon_t
    end
    noise = sqrt(sigma_sq) * epsilon_t
    pt = nu + a * pt_1 + b * noiset_1 + noise
    return pt, sigma_sq, noise
end
