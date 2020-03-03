# Product
struct Product{D<:NondeterministicScalar, T, N, A<:AbstractArray{T,N}} <:
       NondeterministicArray{T,N}
    params :: Tuple
    val :: A
    function Product{D}(params; val) where D
        # eltype(D) == eltype(val) || error("distribution and value element types must coincide")
        D == Base.typename(D).wrapper || @warn "distribution domain parameter is inferred from value, the specified type will be ignored"
        new{Base.typename(D).wrapper,
            eltype(val), ndims(val), typeof(val)}(params, val)
    end
end

Product{D}(θs::AbstractArray{T}...) where {D,T} =
    Product{D}(xs; val = (forgetful ∘ D).(θs...))

Product{Normal}(μs::CuArray{T}, σs::CuArray{T}) where T =
    Product{Normal{T}}((μs, σs); val = μs .+ σs .* CuArrays.randn(T, size(μs)))

# fallback CuArray constructor (sampling on cpu and moving to the device)
Product{D}(θs::CuArray{T}...) where {D,T} =
    Product{D{T}}(θs; val = CuArray((forgetful ∘ D).(Array.(θs)...)))


loglikelihood(x, ::Type{<:Product{D}}, θs...) where D =
    sum(loglikelihood.(x, [D], θs...))

_loglikelihood(x, ::Type{<:Product{D}}, θs...) where D =
    sum(_loglikelihood.(x, CuArray([D]), θs...))

# # Mixture. I don't really know if this belongs here, because it has "hidden variables" (the component of the mixture that the value was sampled from). Thus, computing loglikelihood is funny...
# struct Coproduct{D,N,T} <: NondeterministicScalar{T}
#     val :: T
#     Coproduct{N}(;val::NondeterministicScalar{T}) where {N,T} =
#         new{typeof(val),N,eltype(val)}(val)
# end
#
# function Coproduct{D}(θ::SVector{N}...; p::SVector{N,F}) where {D,N,F}
#     i = Categorical(p)
#     Coproduct{N}(val = D(map(x -> x[i], θ)...))
# end


# Dirichlet
struct Dirichlet{N,F<:Real} <: NondeterministicArray{F,1}
    params :: Tuple
    val :: SVector{N,F}
    Dirichlet(params; val) = new{length(val),eltype(val)}(params, val)
end

function Dirichlet(α::SVector{N,F}) where {N,F}
    p = Product{Gamma}(α, [one(F)])
    Dirichlet((α,); val = p/sum(p))
end

Dirichlet{N}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N,F}(α::SVector{N,F}) where {N,F} = Dirichlet(α)
Dirichlet{N}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))
Dirichlet{N,F}(α::F) where {N,F} = Dirichlet(SVector(ntuple(_->α, N)))


# loglikelihood(p::Dirichlet, α) =
# _loglikelihood(p::Dirichlet, α) =
