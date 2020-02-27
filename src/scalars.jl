# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)

# Discrete Nondeterministics
for D in [:Categorical, :Poisson]
    @eval struct $D{T<:Integer} <: NondeterministicScalar{T}
        val :: T
        $D(;val) = new{typeof(val)}(val)
        $D{T}(;val) where T = new{T}(val)
    end
end

# Continuous Nondeterministics
for D in [:Uniform, :Normal, :Exponential, :Gamma]
    @eval struct $D{T<:AbstractFloat} <: NondeterministicScalar{T}
        val :: T
        $D(;val) = new{typeof(val)}(val)
        $D{T}(;val) where T = new{T}(val)
    end
end


##########################
# Sampling and likelihood
##########################

# Categorical
function Categorical(p::AbstractVector{F}) where F<:AbstractFloat
    draw = rand(F)
    cp = zero(F)
    i = 0
    while cp < draw && i < length(p)
        cp += p[i +=1]
    end
    Categorical(val = max(i,1))
end

loglikelihood(c::Categorical, p::AbstractVector{F}) where F<:AbstractFloat =
    ifelse(1 ≤ c.val ≤ length(p), @inbounds log(p[c.val]), -F(Inf))

_loglikelihood(c::Categorical, p::AbstractVector{F}) where F<:AbstractFloat =
    ifelse(1 ≤ c.val ≤ length(p), @inbounds CUDAnative.log(p[c.val]), -F(Inf))


# Poisson
# TODO eliminar dependencia de StatsFuns:
Poisson(λ::F) where F<:AbstractFloat =
    Poisson(val = convert(Int, poisinvcdf(λ, rand())))

loglikelihood(p::Poisson, λ::F) where F<:AbstractFloat = poislogpdf(λ, p.val)

function _loglikelihood(p::Poisson, λ::F) where F<:AbstractFloat
    x = convert(F, p.val)
    iszero(λ) && return ifelse(iszero(x), zero(F), -F(Inf))
    x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(F))
end


# Uniform
Uniform(a::F, b::F) where F = Uniform(val = a + (b - a)rand(F))
Uniform{F}(a::F, b::F) where F = Uniform(val = a + (b - a)rand(F))

loglikelihood(u::Uniform{F}, a::F, b::F) where F =
    ifelse(a ≤ u.val ≤ b, -log(b - a), -F(Inf))

_loglikelihood(u::Uniform{F}, a::F, b::F) where F =
    ifelse(a ≤ u.val ≤ b, -CUDAnative.log(b - a), -F(Inf))


# Normal
Normal(μ::F, σ::F) where F = Normal(val = μ + σ * randn(F))
Normal{F}(μ::F, σ::F) where F = Normal(val = μ + σ * randn(F))

function loglikelihood(x::Normal{F}, μ::F, σ::F) where F
    iszero(σ) && return ifelse(x == μ, Inf, -Inf)
    -(((x.val - μ) / σ)^2 + log2π)/2 - log(σ)
end

function _loglikelihood(x::Normal{F}, μ::F, σ::F) where F
    iszero(σ) && return ifelse(x == μ, F(Inf), -F(Inf))
    -(((x.val - μ) / σ)^2 + F(log2π))/2 - CUDAnative.log(σ)
end


# Exponential
Exponential(λ::F) where F = Exponential(val = λ * randexp(F))
Exponential{F}(λ::F) where F = Exponential(val = λ * randexp(F))

loglikelihood(x::Exponential{F}, λ::F) where F =
    ifelse(x.val < zero(F), -F(Inf), log(λ) - λ * x)

_loglikelihood(x::Exponential{F}, λ::F) where F =
    ifelse(x.val < zero(F), -F(Inf), CUDAnative.log(λ) - λ * x)


# Gamma
function _MarsagliaTsang2000(α::F) where F
    d = α - one(F)/3
    c = one(F) / sqrt(9d)
    while true
        x = randn(F)
        v = (one(F) + c*x)^3
        while v < zero(F)
            x = randn(F)
            v = (1 + c*x)^3
        end
        u = rand(F)
        # u < one(F) - F(0.0331)x^4 && return d*v
        log(u) < x^2/2 + d*(one(F) - v + log(v)) && return d*v
    end
end

function Gamma(α::F, θ::F) where F
    if α < one(F) # use the γ(1+α)*U^(1/α) trick from Marsaglia and Tsang (2000)
        x = θ * _MarsagliaTsang2000(α + 1) # Gamma(α + 1, θ)
        e = randexp(F)
        return Gamma(val = x * exp(-e / α))
    elseif α == one(F)
        return Gamma(val = θ * randexp(F))
    else
        return Gamma(val = θ * _MarsagliaTsang2000(α))
    end
end

# TODO eliminar dependencia de StatsFuns:
loglikelihood(x::Gamma{F}, α::F, θ::F) where F = gammalogpdf(α, θ, x.val)

_loglikelihood(x::Gamma{F}, α::F, θ::F) where F =
    -CUDAnative.lgamma(k) - α*CUDAnative.log(θ) + (α-one(F))*CUDAnative.log(x.val) - x.val/θ
