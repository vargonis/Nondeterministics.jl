struct Product{D, N, A <: AbstractArray{D,N}} <: AbstractArray{D,N}
    val :: A
end

Base.size(p::Product) = size(p.val)
Base.getindex(p::Product, is::Int...) = getindex(p.val, is...)
Base.setindex!(p::Product, is::Int...) = setindex!(p.val ,is...)
Base.show(io::IO, p::Product) = show(io, p.val)

function Product{D}(xs::AbstractArray{T,N}...) where {D,T,N}
    val = D.(xs...)
    Product{D,N,typeof(val)}(val)
end

# Mixtures:
# struct Coproduct{D, N, A <: } <:

# @inline function logpdf(p::Product, xs::AbstractArray) # D<:Sampleable{X}
#     sum(logpdf(eltype(p)).(p.f.(p.params...), xs))
# end
