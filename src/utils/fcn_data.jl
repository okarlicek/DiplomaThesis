include("../models/franke_and_westerhoff.jl")
include("../models/arma_garch.jl")
include("../models/ar.jl")


function generate_series(setup::Dict, obs::Int, burn::Int, theta::Array{Float64,1}, returns::Bool=false)
    # retrieve model's implementation and generate data
    if setup["mod"]["model"] == "fw2012_5p"
        data = generateFWModel5p(obs, burn, theta, setup["mod"]["fixed_params"], returns=returns)
    elseif setup["mod"]["model"] == "ARMAGARCH"
        data = generateARMAGARCH(obs, burn, theta, setup["mod"]["fixed_params"])
    elseif setup["mod"]["model"] == "AR2"
        data = generateAR2(obs, burn, theta, setup["mod"]["fixed_params"])
    else
        error("The chosen model is not implemented.")
    end
    return data
end


function generate_model_step(setup::Dict, step_index::Int, data::Array{Float64,1}, simulated_values::Array{Float64,2},
                             noise::Array{Float64}, theta::Array{Float64,1}, latent::Union{Nothing, Array{Float64,1}}, past_noise::Array{Float64,1})
    # generate trajectories utilized in loglikelihood approximation
    traj = setup["opt"]["traj"]
    fixed_params = setup["mod"]["fixed_params"]
    pt = 0.

    if setup["mod"]["model"] == "fw2012_5p"
        if (step_index < 3) return false end
        for j in axes(simulated_values)[2]
            # trajectory initialization
            lat, pt_1, pt_2 = latent[j], data[step_index - 1], data[step_index - 2]
            for t in 1:traj
                pt, lat = FWstep5p(theta, fixed_params, pt_1, pt_2, lat, noise[step_index+t-1, j,1], noise[step_index+t-1, j,2])
                pt_2 = pt_1
                pt_1 = pt
                simulated_values[t, j] = pt
            end
        end
    elseif setup["mod"]["model"] == "ARMAGARCH"
        if (step_index < 3) return false end

        fit_pt, _, _ = ARMAGARCHstep(theta, fixed_params, data[step_index - 2], past_noise[1], mean(latent), 0.)
        past_noise[1] = data[step_index - 1] - fit_pt  # estimating the noise in past step

        for j in axes(simulated_values)[2]
            # trajectory initialization
            pt_1, noiset_1, sigmasqt_1 = data[step_index - 1], past_noise[1], latent[j]
            for t in 1:traj
                pt, sigmasqt_1, noiset_1 = ARMAGARCHstep(theta,  fixed_params, pt_1, noiset_1, sigmasqt_1, noise[step_index+t-1, j])
                pt_1 = pt
                simulated_values[t, j] = pt
            end
        end
    elseif setup["mod"]["model"] == "AR2"
        if (step_index < 3) return false end
        for j in axes(simulated_values)[2]
            # trajectory initialization
            pt_1, pt_2 = data[step_index - 1], data[step_index - 2]
            for t in 1:traj
                pt = AR2step(theta, fixed_params, pt_1, pt_2, noise[step_index+t-1, j])
                pt_2 = pt_1
                pt_1 = pt
                simulated_values[t, j] = pt
            end
        end
    else
        error("The chosen model is not implemented")
    end
    return true
end


function generate_latent_vector(setup::Dict, theta::Array{Float64,1})
    # burn-in periods of the model to obtain vector of latent variables
    burn = setup["opt"]["burn"]
    fixed_params = setup["mod"]["fixed_params"]

    if setup["mod"]["model"] == "fw2012_5p"
        latent = zeros(setup["opt"]["sim"])

        for i in eachindex(latent)
            pt_1, pt_2, at_2 = 0., 0., 0.5
            for j in 1:burn
                pt_temp, at_2 = FWstep5p(theta, fixed_params, pt_1, pt_2, at_2)
                pt_2 = pt_1
                pt_1 = pt_temp
            end
            latent[i] = at_2
        end
    elseif setup["mod"]["model"] == "ARMAGARCH"
        latent = zeros(setup["opt"]["sim"])

        for i in eachindex(latent)
            pt_1, noiset_1, sigmasqt_1 = 0., 0., 0.
            for t in 1:burn
                pt_1, sigmasqt_1, noiset_1 = ARMAGARCHstep(theta,  fixed_params, pt_1, noiset_1, sigmasqt_1)
            end
            latent[i] = sigmasqt_1
        end
    else
        latent = nothing
    end
    return latent
end
