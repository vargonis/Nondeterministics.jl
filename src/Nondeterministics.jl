module Nondeterministics

using CUDAnative
using CuArrays: @cufunc
using StatsFuns: poisinvcdf, poislogpdf

export Nondeterministic
export Categorical, Categorical16, Categorical32, Categorical64,
       Poisson, Poisson16, Poisson32, Poisson64,
       Uniform, Uniform16, Uniform32, Uniform64,
       Normal, Normal16, Normal32, Normal64
       # Exponential, Exponential16, Exponential32, Exponential64
       # Gamma, Gamma16, Gamma32, Gamma64,
# export Product, Coproduct
export loglikelihood


abstract type Nondeterministic{X} end

include("primitives.jl")
# include("constructions.jl")


end # module
