# Franke and Westerhoff, 2012

function generateFWModel5p(n_observ::Int, burn_in::Int, theta::Array{Float64,1}, fixed_params::Array{Float64,1}, a_zero::Float64=0.5; returns::Bool=false)
    pseudo_data = zeros(burn_in + n_observ)
    a = a_zero
    for i in 3:(n_observ + burn_in)
        p, a = FWstep5p(theta, fixed_params, pseudo_data[i-1], pseudo_data[i-2], a)
        pseudo_data[i] = p
    end

    if returns
        pseudo_data[2:end] = pseudo_data[2:end] - pseudo_data[1:end-1]
    end

    return pseudo_data[burn_in + 1: end]
end


function FWstep5p(theta::Array{Float64,1}, fixed_params::Array{Float64,1}, pt_1::Float64, pt_2::Float64, 
                    at_2::Float64, epsilon_f::Union{Nothing, Float64}=nothing, epsilon_c::Union{Nothing, Float64}=nothing)
    ## SETUP ##
    mu = fixed_params[1] # sensitivity of price to demands
    sigma_f = fixed_params[2] # std. dev of fundamentalist's excess demand, large *5 small /5
    sigma_c = fixed_params[3] # std. dev of chartist's excess demand
    p_fund = fixed_params[4] # fundamental price
    beta = fixed_params[5]
    phi = theta[1] # sensitivity of fundamentalist's demand to price change
    chi = theta[2] # sensitivity of chartist's demand to price change
    alpha_0 = theta[3] # base attraction to fundamentalist
    alpha_n = theta[4] # herding
    alpha_p = theta[5] # misalignment 


    if epsilon_f === nothing
        epsilon_f = randn()
    end
    if epsilon_c === nothing
        epsilon_c = randn()
    end

    # fundamentalist and chartist population equations
    nft_1 = 1 / (1 + exp(- beta * at_2))
    nct_1 = 1 - nft_1
    # excess demands equations
    dft_1 = phi * (p_fund - pt_1) + (epsilon_f * sigma_f)
    dct_1 = chi * (pt_1 - pt_2) + (epsilon_c * sigma_c)
    # price equation
    pt = pt_1 + mu * (nft_1 * dft_1 + nct_1 * dct_1)
    # HPM equation
    at_1 = alpha_n * (nft_1 - nct_1) + alpha_0 + alpha_p * ((pt_1 - p_fund) ^ 2) 
    return pt, at_1
end
