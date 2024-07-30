using Distributed
addprocs(32)
include("../src/main.jl")


mod_set = Dict("model" => "AR2", # model name
               "obs" => 3000, # number of observations
               "burn" => 200, # burn-in period length
               "cali" => [0.2, -0.9], # model calibration; psi_1, psi_2
               "cons" => [(-1., 1.), (-1., 1.)], # search constraints
               "fixed_params" => [sqrt(0.1)]) # sigma

# optimisation setup [default]
opt_set = Dict("type" => "Bayes",
               "particles" => 2000, # number of particles
               "traj" => 1, # trajectory size of the log-likelihood estimation
               "sim" => 400, # number of simulations
               "iter" => 150, # number of SMC iterations
               "burn" => 0, # burn in for latent variable approximation
               )

# full setup dictionary
setup = Dict("mod" => mod_set, # model setup
             "opt" => opt_set) # optimisation setup


for i in 1:100
    path = "./results/ar2_bayes/ar2_bayes_$i.jld"
    seed = i

    @time results = main(setup, seed, path)
end
