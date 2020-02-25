Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)


macro nondeterministic(Type, N, Name)
    NameN = Symbol(Name, N)
    TypeN = Symbol(Type, N)
    gencode = quote
        primitive type $NameN <: Nondeterministic{$TypeN} $N end
        Base.show(io::IO, x::$NameN) = show(io::IO, reinterpret($TypeN, x))
        Base.:(-)(x::$NameN) = reinterpret($NameN, -(reinterpret($TypeN,x)))
    end
    for op in [:(+), :(-), :(*), :(/), :(÷), :(\), :(^), :(%)]
        push!(gencode.args,
            :(@eval Base.$op(x::$NameN, y::$NameN) =
            reinterpret($NameN, $op(reinterpret($TypeN,x), reinterpret($TypeN,y)))))
    end
    # En realidad, suma de algunos de estos produce otra cosa.... Y similarmente con las otras operaciones. Pensarlo!
    gencode
end


for N in [16, 32, 64]
    @eval @nondeterministic Int $N Categorical
    @eval @nondeterministic Int $N Poisson
    @eval @nondeterministic Float $N Uniform
    @eval @nondeterministic Float $N Normal
    @eval @nondeterministic Float $N Exponential
    @eval @nondeterministic Float $N Gamma
end

for D in [:Categorical, :Poisson, :Uniform, :Normal, :Exponential, :Gamma]
    @eval const $D = $(Symbol(D,64))
end


# Categorical
for (C,I) in [(:Categorical16,:Int16), (:Categorical32,:Int32),
              (:Categorical64,:Int64)]
    @eval function $C(p::F...) where F<:AbstractFloat
        n = length(p)
        draw = rand(F)
        cp = zero(F)
        i = 0
        while cp < draw && i < n
            cp += p[i +=1]
        end
        max(i, 1)
    end

    @eval loglikelihood(i::$C, p::F...) where F<:AbstractFloat =
        ifelse(1 ≤ i ≤ length(p), @inbounds log(p[i]), -F(Inf))

    @eval @cufunc loglikelihood(i::$C, p::F...) where F<:AbstractFloat =
        ifelse(1 ≤ i ≤ length(p), @inbounds CUDAnative.log(p[i]), -F(Inf))
end


# Poisson
for (P,I) in [(:Poisson16,:Int16), (:Poisson32,:Int32), (:Poisson64,:Int64)]
    @eval $P(λ::F) where F<:AbstractFloat =
        reinterpret($P, convert($I, poisinvcdf(λ, rand())))

    @eval loglikelihood(i::$P, λ::F) where F<:AbstractFloat = poislogpdf(λ, i)

    @eval @cufunc loglikelihood(i::$P, p::F...) where F<:AbstractFloat = begin
        x = convert(F, reinterpret($I, i))
        iszero(λ) && return ifelse(iszero(x), zero(F), -F(Inf))
        x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(F))
    end
end


# Uniform
for (U,F) in [(:Uniform16,:Float16), (:Uniform32,:Float32), (:Uniform64,:Float64)]
    @eval $U(a::$F, b::$F) = reinterpret($U, a + (b-a) * rand($F))

    @eval function loglikelihood(x::$U, θ::Tuple{$F,$F})
        a, b = θ
        a ≤ reinterpret($F,x) ≤ b ? -log(b - a) : -$F(Inf)
    end

    @eval @cufunc loglikelihood(x::$U, θ::Tuple{$F,$F}) = begin
        a, b = θ
        a ≤ reinterpret($F,x) ≤ b ? -CUDAnative.log(b - a) : -$F(Inf)
    end
end


# Normal
for (N,F) in [(:Normal16,:Float16), (:Normal32,:Float32), (:Normal64,:Float64)]
    @eval $N(μ::$F, σ::$F) = reinterpret($N, μ + σ * randn($F))

    @eval function loglikelihood(x::$N, θ::Tuple{$F,$F})
        μ, σ = θ
        iszero(σ) && return ifelse(x == reinterpret($N, μ), Inf, -Inf)
        -(((reinterpret($F,x) - μ) / σ)^2 + log2π)/2 - log(σ)
    end

    @eval function _loglikelihood(x::$N, θ::Tuple{$F,$F})
        μ, σ = θ
        iszero(σ) && return ifelse(x == reinterpret($N, μ), Inf, -Inf)
        -(((reinterpret($F,x) - μ) / σ)^2 + log2π)/2 - CUDAnative.log(σ)
    end

    @eval @cufunc loglikelihood(x::$N, θ::Tuple{$F,$F}) = _loglikelihood(x, θ)
end


# Exponential


# Gamma
# @which rand(Distributions.GLOBAL_RNG, Gamma())
#
# function rand(rng::AbstractRNG, d::Gamma{T}) where T
#     if shape(d) < 1.0
#         # TODO: shape(d) = 0.5 : use scaled chisq
#         return rand(rng, GammaIPSampler(d))
#     elseif shape(d) == 1.0
#         return rand(rng, Exponential{T}(d.θ))
#     else
#         return rand(rng, GammaGDSampler(d))
#     end
# end
