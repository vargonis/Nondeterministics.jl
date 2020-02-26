struct Product{T<:Real, N, A<:AbstractArray{<:Real,N}} <:
       NondeterministicArray{T,N}
    val :: A
    Product(;val) = new{eltype(eltype(val)),ndims(val),typeof(val)}(val)
end

Base.size(p::Product) = size(p.val)
Base.getindex(p::Product, is::Int...) = getindex(p.val, is...)
Base.setindex!(p::Product, is::Int...) = setindex!(p.val ,is...)

Product{D}(xs::AbstractArray{T,N}...) where {D,T,N} = Product(val = D.(xs...))
Product{Normal}(μs::CuArray{T,N}, σs::CuArray{T,N}) where {T,N} =
    Product(val = μs .+ σs .* CuArrays.randn(T, size(μs)))

# Mixtures:
# struct Coproduct{D, N, A <: } <:


# struct Dirichlet{T<:Real,N} <: NondeterministicArray{T,N}

# @inline function logpdf(p::Product, xs::AbstractArray) # D<:Sampleable{X}
#     sum(logpdf(eltype(p)).(p.f.(p.params...), xs))
# end
