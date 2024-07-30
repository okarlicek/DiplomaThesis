using Distributed
addprocs(32)
include("../src/main.jl")


mod_set = Dict("model" => "ARMAGARCH", # model name
               "obs" => 3000, # number of observations
               "burn" => 200, # burn-in period length
               "cali" => [0.0, 0.7, 0.1, 0.1, 0.3, 0.001], # model calibration; nu, a, b, alpha, beta, omega
               "cons" => [(-1., 1.), (-1., 1.), (-1., 1.), (0., 1.), (0., 1.), (0., 1.)], # search constraints
               "fixed_params" => Float64[])

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
    path = "./results/armagarch_npmsle/armagarch_npmsle_$i.jld"
    seed = i

    @time results = main(setup, seed, path)
end
