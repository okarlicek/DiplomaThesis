# AR(2)


function generateAR2(n_observ::Int, burn_in::Int, theta::Array{Float64,1}, fixed_params::Array{Float64,1})
    pseudo_data = zeros(burn_in + n_observ)

    for i in 3:(n_observ + burn_in)
        p = AR2step(theta, fixed_params, pseudo_data[i-1], pseudo_data[i-2])
        pseudo_data[i] = p
    end
    return pseudo_data[burn_in + 1: end]
end


function AR2step(theta::Array{Float64, 1}, fixed_params::Array{Float64,1}, pt_1::Float64, pt_2::Float64, epsilon_t::Union{Nothing, Float64}=nothing)
    psi_1 = theta[1]
    psi_2 = theta[2]
    sigma = fixed_params[1]

    if epsilon_t === nothing
        epsilon_t = randn()
    end
    
    epsilon_t = epsilon_t * sigma

    pt = psi_1 * pt_1 + psi_2 * pt_2 + epsilon_t
    return pt
end
