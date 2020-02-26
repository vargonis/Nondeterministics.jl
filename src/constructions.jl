struct Product{T, D<:NondeterministicScalar{T},
               N, A<:AbstractArray{D,N}} <: NondeterministicArray{T,N}
    val :: A
    Product(;val) = new{eltype(eltype(val)),eltype(val),ndims(val),typeof(val)}(val)
end

Base.size(p::Product) = size(p.val)
Base.getindex(p::Product, is::Int...) = getindex(p.val, is...)
Base.setindex!(p::Product, is::Int...) = setindex!(p.val ,is...)

Product{D}(xs::AbstractArray{T,N}...) where {D,T,N} = Product(val = D.(xs...))

# Mixtures:
# struct Coproduct{D, N, A <: } <:


# struct Dirichlet{T<:Real,N} <: NondeterministicArray{T,N}

# @inline function logpdf(p::Product, xs::AbstractArray) # D<:Sampleable{X}
#     sum(logpdf(eltype(p)).(p.f.(p.params...), xs))
# end
