# Product
struct Product{D<:NondeterministicScalar, N, A<:AbstractArray{D,N}} <:
       NondeterministicArray{D,N}
    val :: A
    Product(;val) = new{eltype(val),ndims(val),typeof(val)}(val)
end

Product{D}(xs::AbstractArray{T}...) where {D,T} = Product(val = D.(xs...))
Product{Normal}(μs::CuArray{T}, σs::CuArray{T}) where T =
    Product(val = (x -> Normal(val=x)).(μs .+ σs .* CuArrays.randn(T, size(μs))))

loglikelihood(p::Product, θs...) = sum(loglikelihood.(p.val, θs...))
# _loglikelihood(p::Product, θs...) =


# Mixture. I don't really know if this belongs here, because it has "hidden variables" (the component of the mixture that the value was sampled from). Thus, computing loglikelihood is funny...
struct Coproduct{D,N,T} <: NondeterministicScalar{T}
    val :: T
    Coproduct{N}(;val::NondeterministicScalar{T}) where {N,T} =
        new{typeof(val),N,eltype(val)}(val)
end

function Coproduct{D}(θ::SVector{N}...; p::SVector{N,F}) where {D,N,F}
    i = Categorical(p)
    Coproduct{N}(val = D(map(x -> x[i], θ)...))
end


# Dirichlet
struct Dirichlet{N,F<:Real} <: NondeterministicArray{F,1}
    val :: SVector{N,F}
    Dirichlet(;val) = new{length(val),eltype(val)}(val)
end

function Dirichlet(α::SVector{N,F}) where {N,F}
    p = Product{Gamma}(α, [one(F)])
    Dirichlet(val = p/sum(p))
end

Dirichlet{N}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N,F}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))
Dirichlet{N,F}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))


# loglikelihood(p::Dirichlet, α) =
# _loglikelihood(p::Dirichlet, α) =
