
function uniform_bias(lower_bound::Float64, upper_bound::Float64)
    r = rand()
    
    r *= upper_bound - lower_bound
    r += lower_bound
    return r 
end

function initialize_particles(particles::Array{Float64, 2}, setup::Dict)
    for i in axes(particles)[2]

        for j in eachindex(setup["mod"]["cons"])
            lower, upper = setup["mod"]["cons"][j]
            particles[j, i] = uniform_bias(lower, upper)
        end
    end
    return particles
end