using Distributed
addprocs(32)
include("../src/main.jl")


mod_set = Dict("model" => "fw2012_5p", # model name
               "obs" => 3000, # number of observations
               "burn" => 200, # burn-in period length
               "cali" => [0.12, 1.50, - 0.327, 1.79, 18.43], # model calibration; phi, chi, alpha_0, alpha_n, alpha_p
               "cons" => [(0., 3.), (0., 3.), (-1., 1.), (0.5, 4.), (5., 50.)], # search constraints
               "fixed_params" => [0.01, 3.79, 10.435, 0.0, 1.]) # mu, sigma_f, sigma_c, p_fund, beta


# optimisation setup [default]
opt_set = Dict("type" => "SMM",
               "rep" => 96, # number of repetitions
               "simfactor" => 1,  # simulated series length factor
               "inits" => 1, # number of initial points
               "sim" => 400, # number of simulations
               "iter" => 4000, # number of iterations
               "mom_set" => [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # which moments to use
               )

# weighting matrix setup [default] ~ {options}
wgt_set = Dict("method" => "fw2012overlap", # approach to weighting matrix construction ~ {"eye","fw2012","fw2012overlap","fw2012overlapdiag"}
               "bootsize" => 5000) # repetitions for block-bootstrapped weighting matrix

# full setup dictionary
setup = Dict("mod" => mod_set, # model setup
             "opt" => opt_set, # optimisation setup
             "wgt" => wgt_set) # weighting matrix setup


for i in 1:100
    path = "./results/fw5p_large_smm/fw5p_large_smm_$i.jld"
    seed = i

    @time results = main(setup, seed, path)
end
