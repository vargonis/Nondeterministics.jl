module Nondeterministics

using Random
using StatsFuns: log2π, poisinvcdf, poislogpdf, gammalogpdf
using StaticArrays
using CUDAnative
using CuArrays

export NondeterministicScalar, NondeterministicArray
export Categorical, Poisson, Uniform, Normal, Exponential, Gamma
export Product, Coproduct, Dirichlet
export loglikelihood


abstract type NondeterministicScalar{T<:Real} <: Real end
abstract type NondeterministicArray{T,N} <: AbstractArray{T,N} end


for N in [:NondeterministicScalar, :NondeterministicArray]
    @eval Base.promote_rule(::Type{T}, ::Type{<:$N{T}}) where T = T
    @eval Base.promote_rule(::Type{<:$N{T}}, ::Type{<:$N{T}}) where T = T
    @eval (::Type{T})(d::$N{T}) where T = d.val
    @eval Base.show(io::IO, d::$N) = show(io, d.val)
    for op in [:(+), :(-), :(*), :(/), :(÷), :(\), :(^), :(%),
               :(<), :(<=), :(>), :(>=)]
        @eval function Base.$op(x::D, y::D) where {T, D<:$N{T}}
            $op(x.val, y.val)
        end
    end
end


Base.to_index(d::NondeterministicScalar{T}) where T<:Integer = d.val
Base.eltype(::NondeterministicScalar{T}) where T = T
Base.eltype(::Type{<:NondeterministicScalar{T}}) where T = T

Base.size(d::NondeterministicArray) = size(d.val)
Base.getindex(d::NondeterministicArray, args...) = getindex(d.val, args...)
Base.setindex!(d::NondeterministicArray, args...) = setindex!(d.val, args...)
Base.print_array(io::IO, d::NondeterministicArray) = Base.print_array(io, d.val)
Base.eltype(::NondeterministicArray{T,N}) where {T,N} = T
Base.eltype(::Type{<:NondeterministicArray{T,N}}) where {T,N} = T


include("scalars.jl")
include("arrays.jl")

CuArrays.@cufunc loglikelihood(d::NondeterministicScalar, θ...) =
    _loglikelihood(d, θ...)


end # module
