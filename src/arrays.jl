# Dirichlet
struct Dirichlet{N, S<:Real, T<:AbstractVector{S}, P<:AbstractVector{S}} <: NondeterministicArray{S,1}
    params :: P
    val :: T
    function Dirichlet(params::AbstractVector; val)
        length(params) == length(val) || error("Dirichlet: parameters and value must have same length")
        new{length(params), eltype(val), typeof(val), typeof(params)}(params, val)
    end
end

function Dirichlet(α::SVector{N,S}) where {N,S}
    p = Gamma.(α, [one(eltype(S))])
    Dirichlet(α; val = p / sum(p))
end

Dirichlet{N}(α::SVector{N}) where N = Dirichlet(α)
Dirichlet{N,S}(α::SVector{N,S}) where {N,S} = Dirichlet(α)
Dirichlet{N}(α::S) where {N,S} = Dirichlet(SVector(ntuple(_->α, N)))
Dirichlet{N,S}(α::S) where {N,S} = Dirichlet(SVector(ntuple(_->α, N)))


function loglikelihood(d::Dirichlet)
    α = params(d)
    a, b = sum(u -> SVector(u,loggamma(u)), α)
    s = sum((u,v) -> (u-one(F)log(v)), zip(α, d.val))
    s - b + loggamma(a)
end

function _loglikelihood(d::Dirichlet)
    α = params(d)
    a, b = sum(u -> SVector(u,CUDAnative.lgamma(u)), α)
    s = sum((u,v) -> (u-one(F)CUDAnative.log(v)), zip(α, d.val))
    s - b + CUDAnative.lgamma(a)
end
