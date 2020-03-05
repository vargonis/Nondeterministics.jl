# Dirichlet
struct Dirichlet{N,F<:Real} <: NondeterministicArray{F,1}
    params :: Tuple
    val :: SVector{N,F}
    Dirichlet(params; val) = new{length(val),eltype(val)}(params, val)
end

function Dirichlet(α::SVector{N,F}) where {N,F}
    p = Product{Gamma}(α, [one(eltype(F))])
    Dirichlet((α,); val = p/sum(p))
end

Dirichlet{N}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N,F}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))
Dirichlet{N,F}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))


function loglikelihood(x::AbstractVector{F},
                       ::Type{<:Dirichlet},
                       α::AbstractVector{F}) where F<:AbstractFloat
    a, b = sum(u -> SVector(u,loggamma(u)), α)
    s = sum((u,v) -> (u-one(F)log(v)), zip(α,x))
    s - b + loggamma(a)
end

function _loglikelihood(x::AbstractVector{F},
                        ::Type{<:Dirichlet},
                        α::AbstractVector{F}) where F<:AbstractFloat
    a, b = sum(u -> SVector(u,CUDAnative.lgamma(u)), α)
    s = sum((u,v) -> (u-one(F)CUDAnative.log(v)), zip(α,x))
    s - b + CUDAnative.lgamma(a)
end
