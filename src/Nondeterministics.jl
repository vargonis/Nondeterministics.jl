module Nondeterministics

using CUDAnative
using CuArrays: @cufunc
using StatsFuns: poisinvcdf, poislogpdf, log2π

export Nondeterministic
export Categorical, Poisson, Uniform, Normal # Exponential, Gamma
export Product # Coproduct
export loglikelihood


abstract type Nondeterministic{T<:Real} <: Real end

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
Base.promote_rule(::Type{T}, ::Type{<:Nondeterministic{T}}) where T = T
Base.promote_rule(::Type{<:Nondeterministic{T}},
                  ::Type{<:Nondeterministic{T}}) where T = T
(::Type{T})(d::Nondeterministic{T}) where T = d.val
Base.show(io::IO, d::Nondeterministic) = show(io, d.val)
Base.to_index(d::Nondeterministic{T}) where T<:Integer = d.val


include("scalars.jl")
include("constructions.jl")

@cufunc loglikelihood(d::Nondeterministic, θ) = _loglikelihood(d, θ)


end # module
