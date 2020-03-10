# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)
# const log2π = 1.8378770664093454836

# Discrete Nondeterministics
for (D, P) in [(:Categorical, Tuple{Vararg{Real}}),
               (:Poisson, Tuple{Real})]
    @eval struct $D{T<:Integer, P} <: NondeterministicInteger{T}
        params :: P
        val    :: T
        $D(params::$P; val) = new{typeof(val),typeof(params)}(params, val)
        $D{T}(params::$P; val) where T = new{T,typeof(params)}(params, val)
    end
end

# Continuous Nondeterministics
for (D, P) in [(:Uniform, Tuple{Real,Real}),
               (:Normal, Tuple{Real,Real}),
               (:Exponential, Tuple{Real}),
               (:Gamma, Tuple{Real,Real})]
    @eval struct $D{T<:Real, P} <: NondeterministicReal{T}
        params :: P
        val    :: T
        $D(params::$P; val) = new{typeof(val),typeof(params)}(params, val)
        $D{T}(params::$P; val) where T = new{T,typeof(params)}(params, val)
    end
end


##########################
# Sampling and likelihood
##########################

# Categorical
function homogenize(t::Tuple)
    T = promote_type(typeof(t).parameters...)
    Tuple{Vararg{T}}(t)
end

function Categorical(p::Union{AbstractVector{T},Tuple{Vararg{T}}}) where T<:Real
    draw = rand(eltype(T))
    cp = zero(eltype(T))
    i = 0
    while cp < draw && i < length(p)
        cp += p[i+=1]
    end
    tp = p isa Tuple ? homogenize(p) : Tuple(p)
    Categorical(tp; val = max(i,1))
end

function loglikelihood(c::Categorical)
    p = c.params
    ifelse(1 ≤ c.val ≤ length(p), @inbounds log(p[c.val]), -eltype(p)(Inf))
end

function _loglikelihood(c::Categorical)
    p = c.params
    ifelse(1 ≤ c.val ≤ length(p), @inbounds CUDAnative.log(p[c.val]), -eltype(p)(Inf))
end


# Poisson
# TODO eliminar dependencia de StatsFuns:
Poisson(λ::Real) =
    Poisson((λ,); val = convert(Int, poisinvcdf(λ, rand())))

loglikelihood(p::Poisson) = poislogpdf(p.params..., p.val)

function _loglikelihood(p::Poisson)
    λ, = p.params
    T = eltype(λ)
    x = convert(F, p.val)
    iszero(λ) && return ifelse(iszero(x), zero(T), -T(Inf))
    x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(F))
end


# Uniform
Uniform(a::Real, b::Real) = Uniform(promote(a, b)...)
Uniform(a::T, b::T) where T<:Real =
    Uniform((a, b); val = a + (b - a)rand(eltype(T)))

function loglikelihood(u::Uniform)
    a, b = u.params
    ifelse(a ≤ u.val ≤ b, -log(b - a), -typeof(a)(Inf))
end

function _loglikelihood(u::Uniform)
    a, b = promote(u.params...)
    ifelse(a ≤ u.val ≤ b, -CUDAnative.log(b - a), -typeof(a)(Inf))
end


# Normal
Normal(μ::Real, σ::Real) = Normal(promote(μ, σ)...)
Normal(μ::T, σ::T) where T<:Real =
    Normal((μ, σ); val = μ + σ * randn(eltype(T)))

function loglikelihood(x::Normal)
    μ, σ = x.params
    T = eltype(μ)
    iszero(σ) && return ifelse(x.val == μ, T(Inf), -T(Inf))
    -(((x.val - μ) / σ)^2 + T(log2π))/2 - log(σ)
end

function _loglikelihood(x::Normal)
    μ, σ = x.params
    T = eltype(μ)
    iszero(σ) && return ifelse(x.val == μ, T(Inf), -T(Inf))
    -(((x.val - μ) / σ)^2 + T(log2π))/2 - CUDAnative.log(σ)
end


# Exponential
Exponential(λ::Real) =
    Exponential((λ,); val = λ * randexp(eltype(T)))

function loglikelihood(x::Exponential)
    λ, = x.params
    T = eltype(λ)
    ifelse(x.val < zero(T), -T(Inf), log(λ) - λ * x.val)
end

function _loglikelihood(x::Exponential)
    λ, = x.params
    T = eltype(λ)
    ifelse(x.val < zero(T), -T(Inf), CUDAnative.log(λ) - λ * x.val)
end


# Gamma
function _MarsagliaTsang2000(α::Real)
    d = α - one(T)/3
    c = one(T) / sqrt(9d)
    while true
        x = randn(T)
        v = (one(T) + c*x)^3
        while v < zero(T)
            x = randn(T)
            v = (one(T) + c*x)^3
        end
        u = rand(T)
        # u < one(F) - F(0.0331)x^4 && return d*v
        log(u) < x^2/2 + d*(one(T) - v + log(v)) && return d*v
    end
end

Gamma(α::Real, θ::Real) = Gamma(promote(α, θ)...)
function Gamma(α::T, θ::T) where T<:Real
    if α < one(eltype(T)) # use the γ(1+α)*U^(1/α) trick from Marsaglia and Tsang (2000)
        x = θ * _MarsagliaTsang2000(α + one(eltype(T))) # Gamma(α + 1, θ)
        e = randexp(eltype(T))
        return Gamma((α, θ); val = x * exp(-e / α))
    elseif α == one(eltype(F))
        return Gamma((α, θ); val = θ * randexp(eltype(T)))
    else
        return Gamma((α, θ); val = θ * _MarsagliaTsang2000(α))
    end
end

# TODO eliminar dependencia de StatsFuns:
function loglikelihood(x::Gamma)
    α, θ = x.params
    gammalogpdf(α, θ, x.val)
end

function _loglikelihood(x::Gamma)
    α, θ = x.params
    -CUDAnative.lgamma(α) - α*CUDAnative.log(θ) + (α-one(α))*CUDAnative.log(x) - x/θ
end
