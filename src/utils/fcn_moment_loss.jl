"""
    moment_loss_function(theta, setup, moments_emp, weights, simlen)

Calculate the J value computed as the weighted difference between empirical 
moments and average simulated moments.

# Arguments
- `theta::Array{Float64}`: parameter values
- `setup::Dict`: full setup dictionary
- `moments_emp::Array`: array of empirical moments
- `weights`: SMM weighting matrix
- `simlen::Int`: length of the simulated series
"""
function moment_loss_function(theta::Array{Float64, 1}, setup::Dict, moments_emp::Array{Float64, 1}, weights::Array{Float64, 2}, simlen::Int)
    moments_sim = zeros(sum(setup["opt"]["mom_set"]), setup["opt"]["sim"]) # array of simulated moments

    for i in 1:setup["opt"]["sim"]
        # simulate time series and calculate simulated moments
        data = generate_series(setup, simlen, setup["mod"]["burn"], theta, true)
        moments_sim[:,i] = calculate_moment_vector(data, setup)
    end

    # calculate J value
    moments_diff = mean(moments_sim, dims=2) - moments_emp
    obj = transpose(moments_diff) * weights * moments_diff 
    
    if isnan(obj[1])
        obj[1] = prevfloat(Inf)
    end

    return obj[1]
end
