
function generate_random_noise(setup::Dict)
    if setup["mod"]["model"] == "fw2012_5p"
        noise = randn(setup["mod"]["obs"], setup["opt"]["sim"], 2)
    elseif setup["mod"]["model"] == "ARMAGARCH"
        noise = randn(setup["mod"]["obs"], setup["opt"]["sim"])
    elseif setup["mod"]["model"] == "AR2"
        noise = randn(setup["mod"]["obs"], setup["opt"]["sim"])
    else
        error("The chosen model is not implemented.")
    end
    return noise
end
