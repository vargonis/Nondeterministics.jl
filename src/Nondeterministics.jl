module Nondeterministics

using Random
using StatsFuns: log2π, poisinvcdf, poislogpdf, gammalogpdf
using CUDAnative
using CuArrays

export Nondeterministic
export Categorical, Poisson, Uniform, Normal # Exponential, Gamma
export Product # Coproduct
export loglikelihood


abstract type NondeterministicScalar{T<:Real} <: Real end
abstract type NondeterministicArray{T,N} <: AbstractArray{T,N} end

# Esto es facil pero matematicamente muy inadecuado, y ademas la implementacion de la promocion por defecto (mediante construccion) viola la semantica de Nondeterministics:
# Base.promote_rule(::Type{D}, ::Type{T}) where {T<:Real, D<:Nondeterministic{T}} = D

# for op in [:(+), :(-), :(*), :(/), :(÷), :(\), :(^), :(%)]
#     @eval function Base.$op(x::D, y::D) where {T, D<:Nondeterministic{T}}
#         D($op(x.val, y.val))
#     end
# end
# En realidad, suma de algunos de estos produce otra cosa.... Y similarmente con las otras operaciones.
# Creo que me conviene perder la informacion del modelo no deterministico al hacer operaciones aritmeticas.
# Quizas a futuro encuentre uso para una version de las operaciones aritmeticas que preserven tal informacion (cuando eso sea posible).
# En consecuencia:
for N in [:NondeterministicScalar, :NondeterministicArray]
    @eval Base.promote_rule(::Type{T}, ::Type{<:$N{T}}) where T = T
    @eval Base.promote_rule(::Type{<:$N{T}}, ::Type{<:$N{T}}) where T = T
    @eval (::Type{T})(d::$N{T}) where T = d.val
    @eval Base.show(io::IO, d::$N) = show(io, d.val)
end

Base.to_index(d::NondeterministicScalar{T}) where T<:Integer = d.val
Base.eltype(::NondeterministicScalar{T}) where T = T
Base.eltype(::Type{<:NondeterministicScalar{T}}) where T = T

Base.print_array(io::IO, d::NondeterministicArray) = Base.print_array(io, d.val)
Base.eltype(::NondeterministicArray{T,N}) where {T,N} = T
Base.eltype(::Type{<:NondeterministicArray{T,N}}) where {T,N} = T


include("scalars.jl")
include("constructions.jl")

CuArrays.@cufunc loglikelihood(d::NondeterministicScalar, θ...) =
    _loglikelihood(d, θ...)


end # module
