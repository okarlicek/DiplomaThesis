
"""
    calculate_moment_vector(data, mom_set, data_longmem)

Calculate moments of a time series for given model and remove those not included in the moment 
set. If `data_longmem` is populated, long-memory moments are calculated using 
this time series instead.

# Arguments
- `data::Array{Float64,1}`: array of observations
- `setup::Dict`: setup of the problem
- `data_longmem=nothing`: array of observations for long-memory moments
"""
function calculate_moment_vector(data::Array{Float64,1}, setup::Dict, data_longmem=nothing)
    if setup["mod"]["model"] == "fw2012_5p"
        moments = FWmoments(data, setup["opt"]["mom_set"], data_longmem)
    elseif setup["mod"]["model"] == "ARMAGARCH"
        moments = ARMAGARCHmoments(data, setup["opt"]["mom_set"], data_longmem)
    elseif setup["mod"]["model"] == "AR2"
        moments = AR2moments(data, setup["opt"]["mom_set"], data_longmem)
    else
        error("The chosen model is not implemented.")
    end
    return moments
end


"""
    FWmoments(data, mom_set, data_longmem)

Calculate moments for Franke and Westerhoff model.

# Arguments
- `data::Array{Float64,1}`: array of observations
- `mom_set::Array`: moment set
- `data_longmem=nothing`: array of observations for long-memory moments
"""
function FWmoments(data::Array{Float64,1}, mom_set::Array, data_longmem=nothing)
    moments = zeros(22) # array of calculated moments

    # calculate autocorrelations of raw returns
    acf = autocor(data, 0:3)

    # calculate autocorrelations of absolute returns
    absdata = abs.(data)
    absacf = autocor(absdata, 0:101)

    # calculate autocorrelations of squared returns
    sqrdata = data.^2
    sqracf = autocor(sqrdata, 0:26)

    # FW ~ Franke & Westerhoff (2012), CL ~ Chen & Lux (2018)
    moments[1] = var(data) # variance of raw returns [CL]
    moments[2] = kurtosis(data)+3 # kurtosis of raw returns [CL]
    moments[3] = acf[2,1] # 1st lag AC of raw returns [FW,CL]
    moments[4] = acf[3,1] # 2nd lag AC of raw returns
    moments[5] = acf[4,1] # 3rd lag AC of raw returns

    moments[6] = mean(absdata) # mean of absolute returns [FW]
    moments[7] = hill(absdata, 2.5) # Hill estimator (2.5% of the right tail) of absolute returns
    moments[8] = hill(absdata, 5) # Hill estimator (5% of the right tail) of absolute returns [FW]

    moments[9] = mean(absacf[2:3,1]) # 1st lag AC of absolute returns [FW,CL]
    moments[10] = mean(absacf[5:7,1]) # 5th lag AC of absolute returns [FW,CL]
    moments[11] = mean(absacf[10:12,1]) # 10th lag AC of absolute returns [FW,CL]
    moments[12] = mean(absacf[15:17,1]) # 15th lag AC of absolute returns [CL]
    moments[13] = mean(absacf[20:22,1]) # 20th lag AC of absolute returns [CL]
    moments[14] = mean(absacf[25:27,1]) # 25th lag AC of absolute returns [FW,CL]
    moments[15] = mean(absacf[50:52,1]) # 50th lag AC of absolute returns [FW]
    moments[16] = mean(absacf[100:102,1]) # 100th lag AC of absolute returns [FW]

    moments[17] = mean(sqracf[2:3,1]) # 1st lag AC of squared returns [CL]
    moments[18] = mean(sqracf[5:7,1]) # 5th lag AC of squared returns [CL]
    moments[19] = mean(sqracf[10:12,1]) # 10th lag AC of squared returns [CL]
    moments[20] = mean(sqracf[15:17,1]) # 15th lag AC of squared returns [CL]
    moments[21] = mean(sqracf[20:22,1]) # 20th lag AC of squared returns [CL]
    moments[22] = mean(sqracf[25:27,1]) # 25th lag AC of squared returns [CL]

    # use alternative series to caculate long-memory moments
    if !isnothing(data_longmem)
        # calculate autocorrelations of absolute returns of long-memory 
        absdata_longmem = abs.(data_longmem)
        absacf_longmem = autocor(absdata_longmem, 0:101)

        # calculate autocorrelations of squared returns
        sqrdata_longmem = data_longmem.^2
        sqracf_longmem = autocor(sqrdata_longmem, 0:26)

        # FW ~ Franke & Westerhoff (2012), CL ~ Chen & Lux (2018)
        moments[11] = mean(absacf_longmem[10:12,1]) # 10th lag AC of absolute returns [FW,CL]
        moments[12] = mean(absacf_longmem[15:17,1]) # 15th lag AC of absolute returns [CL]
        moments[13] = mean(absacf_longmem[20:22,1]) # 20th lag AC of absolute returns [CL]
        moments[14] = mean(absacf_longmem[25:27,1]) # 25th lag AC of absolute returns [FW,CL]
        moments[15] = mean(absacf_longmem[50:52,1]) # 50th lag AC of absolute returns [FW]
        moments[16] = mean(absacf_longmem[100:102,1]) # 100th lag AC of absolute returns [FW]

        moments[19] = mean(sqracf_longmem[10:12,1]) # 10th lag AC of squared returns [CL]
        moments[20] = mean(sqracf_longmem[15:17,1]) # 15th lag AC of squared returns [CL]
        moments[21] = mean(sqracf_longmem[20:22,1]) # 20th lag AC of squared returns [CL]
        moments[22] = mean(sqracf_longmem[25:27,1]) # 25th lag AC of squared returns [CL]
    end

    # remove moments not included in the moment set
    moments_sel = [moments[i] for i in eachindex(mom_set) if mom_set[i] == 1]

    return moments_sel
end


"""
    ARMAGARCHmoments(data, mom_set, data_longmem)

Calculate moments for ARMA(1,1)-GARCH(1,1) model.

# Arguments
- `data::Array{Float64,1}`: array of observations
- `mom_set::Array`: moment set
- `data_longmem=nothing`: array of observations for long-memory moments
"""
function ARMAGARCHmoments(data::Array{Float64,1}, mom_set::Array, data_longmem=nothing)
    moments = zeros(10) # array of calculated moments

    # calculate autocorrelations of raw returns
    acf = autocor(data, 0:3)

    # calculate autocorrelations of absolute returns
    absdata = abs.(data)
    absacf = autocor(absdata, 0:3)

    # calculate autocorrelations of squared returns
    sqrdata = data.^2
    sqracf = autocor(sqrdata, 0:3)

    moments[1] = var(data) # variance of raw returns [CL]
    moments[2] = kurtosis(data)+3 # kurtosis of raw returns [CL]
    moments[3] = skewness(data) # skewness of raw returns
    moments[4] = acf[2,1] # 1st lag AC of raw returns [FW,CL]
    moments[5] = acf[3,1] # 2nd lag AC of raw returns

    moments[6] = mean(absdata) # mean of absolute returns [FW]

    moments[7] = absacf[2,1] # 1st lag AC of absolute returns [FW,CL]
    moments[8] = absacf[3,1] # 2th lag AC of absolute returns [FW,CL]

    moments[9] = sqracf[2,1] # 1st lag AC of squared returns [CL]
    moments[10] = sqracf[3,1] # 2th lag AC of squared returns [CL]

    # remove moments not included in the moment set
    moments_sel = [moments[i] for i in eachindex(mom_set) if mom_set[i] == 1]

    return moments_sel
end


"""
    AR2moments(data, mom_set, data_longmem)

Calculate moments for AR(2) model.

# Arguments
- `data::Array{Float64,1}`: array of observations
- `mom_set::Array`: moment set
- `data_longmem=nothing`: array of observations for long-memory moments
"""
function AR2moments(data::Array{Float64,1}, mom_set::Array, data_longmem=nothing)
    moments = zeros(4) # array of calculated moments

    # calculate autocorrelations of raw returns
    acf = autocor(data, 0:4)

    moments[1] = var(data) # variance of raw returns [CL]

    moments[2] = acf[2,1] # 1st lag AC of raw returns [FW,CL]
    moments[3] = acf[3,1] # 2nd lag AC of raw returns
    moments[4] = acf[4,1] # 3rd lag AC of raw returns

    moments_sel = [moments[i] for i in eachindex(mom_set) if mom_set[i] == 1]

    return moments_sel
end


"""
    hill(data, pct)

Calculate Hill estimator at `pct` of the right tail.

# Arguments
- `data::Array{Float64,1}`: array of observations
- `pct`: percentage of the right tail
"""
function hill(data::Array{Float64,1}, pct::Number)
    k = floor(Int, length(data)/100*pct) # determine number of considered points

    sorted = sort(data, rev=true) # sort data from highest to lowest values
    res = sorted[1:k]/sorted[k+1] # normalize considered points

    return ((1/k)*sum(log.(res)))^-1
end
