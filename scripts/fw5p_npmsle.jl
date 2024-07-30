using Distributed
addprocs(32)
include("../src/main.jl")


mod_set = Dict("model" => "fw2012_5p", # model name
               "obs" => 3000, # number of observations
               "burn" => 200, # burn-in period length
               "cali" => [0.12, 1.50, - 0.327, 1.79, 18.43], # model calibration; phi, chi, alpha_0, alpha_n, alpha_p
               "cons" => [(0., 3.), (0., 3.), (-1., 1.), (0.5, 4.), (5., 50.)], # search constraints
               "fixed_params" => [0.01, 0.758, 2.087, 0.0, 1.]) # mu, sigma_f, sigma_c, p_fund, beta

# optimisation setup [default]
opt_set = Dict("type" => "NPMSLE",
               "rep" => 96, # number of repetitions
               "traj" => 1, # trajectory size of the log-likelihood estimation
               "inits" => 1, # number of initial points
               "sim" => 400, # number of simulations
               "iter" => 4000, # number of iterations
               "burn" => 1000, # burn in for latent variable approximation
               )

# full setup dictionary
setup = Dict("mod" => mod_set, # model setup
             "opt" => opt_set) # optimisation setup


for i in 1:100
    path = "./results/fw5p_npmsle/fw5p_npmsle_$i.jld"
    seed = i

    @time results = main(setup, seed, path)
end
