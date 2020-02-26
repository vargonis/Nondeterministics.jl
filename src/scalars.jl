# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)


for D in [:Categorical, :Poisson]
    @eval struct $D{T<:Integer} <: NondeterministicScalar{T}
        val :: T
        $D(;val) = new{typeof(val)}(val)
    end
end

for D in [:Uniform, :Normal, :Exponential, :Gamma]
    @eval struct $D{T<:AbstractFloat} <: NondeterministicScalar{T}
        val :: T
        $D(;val) = new{typeof(val)}(val)
    end
end


# Categorical
function Categorical(p::NTuple{N,F}) where {N,F<:AbstractFloat}
    draw = rand(F)
    cp = zero(F)
    i = 0
    while cp < draw && i < N
        cp += p[i +=1]
    end
    Categorical(val = max(i,1))
end

loglikelihood(c::Categorical, p::NTuple{N,F}) where {N,F<:AbstractFloat} =
    ifelse(1 ≤ c.val ≤ N, @inbounds log(p[c.val]), -F(Inf))

_loglikelihood(c::Categorical, p::NTuple{N,F}) where {N,F<:AbstractFloat} =
    ifelse(1 ≤ c.val ≤ N, @inbounds CUDAnative.log(p[c.val]), -F(Inf))


# Poisson
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

function loglikelihood(u::Uniform{F}, θ::Tuple{F,F}) where F
    a, b = θ
    a ≤ u.val ≤ b ? -log(b - a) : -F(Inf)
end

function _loglikelihood(u::Uniform{F}, θ::Tuple{F,F}) where F
    a, b = θ
    a ≤ u.val ≤ b ? -CUDAnative.log(b - a) : -F(Inf)
end


# Normal
Normal(μ::F, σ::F) where F = Normal(val = μ + σ * randn(F))
Normal{F}(μ::F, σ::F) where F = Normal(val = μ + σ * randn(F))

function loglikelihood(x::Normal{F}, θ::Tuple{F,F}) where F
    μ, σ = θ
    iszero(σ) && return ifelse(x == μ, Inf, -Inf)
    -(((x.val - μ) / σ)^2 + log2π)/2 - log(σ)
end

function _loglikelihood(x::Normal{F}, θ::Tuple{F,F}) where F
    μ, σ = θ
    iszero(σ) && return ifelse(x == μ, Inf, -Inf)
    -(((x.val - μ) / σ)^2 + log2π)/2 - CUDAnative.log(σ)
end


# Exponential


# Gamma
# @which rand(Distributions.GLOBAL_RNG, Gamma())
#
# function rand(rng::AbstractRNG, d::Gamma{T}) where T
#     if shape(d) < 1.0
#         # TODO: shape(d) = 0.5 : use scaled chisq
#         return rand(rng, GammaIPSampler(d))
#     elseif shape(d) == 1.0
#         return rand(rng, Exponential{T}(d.θ))
#     else
#         return rand(rng, GammaGDSampler(d))
#     end
# end
