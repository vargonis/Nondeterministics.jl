module Nondeterministics

using Random
using StatsFuns: log2π, poisinvcdf, poislogpdf, gammalogpdf
using SpecialFunctions: loggamma
using StaticArrays
using CUDAnative
using CuArrays

# import Base: rand

const _distributions = (
    :Categorical, :Poisson,
    :Uniform, :Normal, :Exponential, :Gamma,
    :Dirichlet
)

export Distribution
export params, logpdf
for s in _distributions
    @eval export $s
end


############################
# Distribution generalities
############################

primitive type Distribution 8 end

const Categorical = reinterpret(Distribution, 0x00)
const Poisson     = reinterpret(Distribution, 0x01)
const Uniform     = reinterpret(Distribution, 0x10)
const Normal      = reinterpret(Distribution, 0x11)
const Exponential = reinterpret(Distribution, 0x12)
const Gamma       = reinterpret(Distribution, 0x13)
const Dirichlet   = reinterpret(Distribution, 0x20)

function Base.show(io::IO, d::Distribution)
    for s in distributions
        d == eval(s) && return print(io, string(s))
    end
    print(io, d)
end

Base.length(::Distribution) = 1
Base.iterate(d::Distribution) = (d, nothing)
Base.iterate(::Distribution, ::Nothing) = nothing

@inline function params(d::Distribution)
    d == Categorical && return Tuple{Vararg{Real}}
    d == Poisson     && return Tuple{Real}
    d == Uniform     && return Tuple{Real,Real}
    d == Normal      && return Tuple{Real,Real}
    d == Exponential && return Tuple{Real}
    d == Gamma       && return Tuple{Real,Real}
    d == Dirichlet   && return Tuple{Vararg{Real}}
end


###############################
# Distribution implementations
###############################

# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)

function homogenize(t::Tuple)
    T = promote_type(typeof(t).parameters...)
    Tuple{Vararg{T}}(t)
end

include("scalars.jl")
include("arrays.jl")

@inline function logpdf(d::Distribution)
    for s in _distributions
        d == eval(s) && return eval(Symbol(:logpdf,s))
    end
end

@inline function random(d::Distribution, params...)
    for s in _distributions
        d == eval(s) && return eval(Symbol(:rand,s))(params...)
    end
end
(d::Distribution)(params...) = random(d, params...)

for s in _distributions
    @eval CuArrays.@cufunc $(Symbol(:logpdf,s))(args...) = $(Symbol(:_logpdf,s))(args...)
end


end # module
