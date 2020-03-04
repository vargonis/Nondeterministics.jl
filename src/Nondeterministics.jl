module Nondeterministics

using Random
using StatsFuns: log2π, poisinvcdf, poislogpdf, gammalogpdf
using SpecialFunctions: loggamma
using StaticArrays
using CUDAnative
using CuArrays

export NondeterministicScalar, NondeterministicArray
export Categorical, Poisson, Uniform, Normal, Exponential, Gamma
export Product, Dirichlet
export loglikelihood


abstract type NondeterministicInteger{T<:Integer} <: Integer end
abstract type NondeterministicReal{T<:Real} <: Real end
abstract type NondeterministicArray{T,N} <: AbstractArray{T,N} end
const NondeterministicScalar = Union{NondeterministicInteger, NondeterministicReal}

for N in [:NondeterministicInteger, :NondeterministicReal, :NondeterministicArray]
    @eval Base.promote_rule(::Type{S}, ::Type{<:$N{T}}) where {S<:Number,T} = T
    @eval Base.promote_rule(::Type{<:$N{S}}, ::Type{<:$N{T}}) where {S,T} = promote_rule(S,T)
    @eval (::Type{T})(d::$N) where T = T(d.val)
    @eval Base.show(io::IO, d::$N) = show(io, d.val)
    for op in [:(+), :(-), :(*), :(/), :(÷), :(\), :(^), :(%),
               :(<), :(<=), :(>), :(>=)]
        @eval function Base.$op(x::D, y::D) where {T, D<:$N{T}}
            $op(x.val, y.val)
        end
    end
    @eval loglikelihood(d::$N) = loglikelihood(d.val, typeof(d), d.params...)
    @eval forgetful(d::$N) = d.val
end

forgetful(xs...) = forgetful.(xs)
forgetful(t::Tuple) = forgetful.(t)
forgetful(x) = x

for N in [:NondeterministicInteger, :NondeterministicReal]
    @eval Base.eltype(::$N{T}) where T = T
    @eval Base.eltype(::Type{<:$N{T}}) where T = T
end

Base.size(d::NondeterministicArray) = size(d.val)
Base.getindex(d::NondeterministicArray, args...) = getindex(d.val, args...)
Base.setindex!(d::NondeterministicArray, args...) = setindex!(d.val, args...)
Base.print_array(io::IO, d::NondeterministicArray) = Base.print_array(io, d.val)
Base.eltype(::NondeterministicArray{T,N}) where {T,N} = T
Base.eltype(::Type{<:NondeterministicArray{T,N}}) where {T,N} = T


include("scalars.jl")
include("arrays.jl")

CuArrays.@cufunc loglikelihood(args...) = _loglikelihood(args...)


end # module
