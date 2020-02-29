# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)

# Discrete Nondeterministics
for D in [:Categorical, :Poisson]
    @eval struct $D{T<:Integer} <: NondeterministicInteger{T}
        params :: Tuple
        val    :: T
        $D(params::Tuple; val) = new{typeof(val)}(params, val)
        $D{T}(params::Tuple; val) where T = new{T}(params, val)
    end
end

# Continuous Nondeterministics
for D in [:Uniform, :Normal, :Exponential, :Gamma]
    @eval struct $D{T<:AbstractFloat} <: NondeterministicReal{T}
        params :: Tuple
        val    :: T
        $D(params::Tuple; val) = new{typeof(val)}(params, val)
        $D{T}(params::Tuple; val) where T = new{T}(params, val)
    end
end


##########################
# Sampling and likelihood
##########################

# Categorical
function Categorical(p::AbstractVector{F}) where F<:AbstractFloat
    draw = rand(F)
    cp = zero(F)
    i = 0
    while cp < draw && i < length(p)
        cp += p[i +=1]
    end
    Categorical((p,); val = max(i,1))
end

loglikelihood(::Type{<:Categorical}, p::AbstractVector{F}, n::Integer) where
             F<:AbstractFloat =
    ifelse(1 ≤ n ≤ length(p), @inbounds log(p[n]), -F(Inf))

_loglikelihood(::Type{<:Categorical}, p::AbstractVector{F}, n::Integer) where
              F<:AbstractFloat =
    ifelse(1 ≤ n ≤ length(p), @inbounds CUDAnative.log(p[n]), -F(Inf))


# Poisson
# TODO eliminar dependencia de StatsFuns:
Poisson(λ::F) where F<:AbstractFloat =
    Poisson((λ,); val = convert(Int, poisinvcdf(λ, rand())))

loglikelihood(n::Integer, ::Type{<:Poisson}, λ::F) where F<:AbstractFloat =
    poislogpdf(λ, n)

function _loglikelihood(n::Integer, ::Type{<:Poisson}, λ::F) where F<:AbstractFloat
    x = convert(F, n)
    iszero(λ) && return ifelse(iszero(x), zero(F), -F(Inf))
    x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(F))
end


# Uniform
Uniform(a::F, b::F) where F = Uniform((a, b); val = a + (b - a)rand(F))
Uniform{F}(a::F, b::F) where F = Uniform(a, b)

loglikelihood(x::F, ::Type{<:Uniform}, a::F, b::F) where F =
    ifelse(a ≤ x ≤ b, -log(b - a), -F(Inf))

_loglikelihood(x::F, ::Type{<:Uniform}, a::F, b::F) where F =
    ifelse(a ≤ x ≤ b, -CUDAnative.log(b - a), -F(Inf))


# Normal
Normal(μ::F, σ::F) where F = Normal((μ, σ); val = μ + σ * randn(F))
Normal{F}(μ::F, σ::F) where F = Normal(μ, σ)

function loglikelihood(x::F, ::Type{<:Normal}, μ::F, σ::F) where F
    iszero(σ) && return ifelse(x == μ, F(Inf), -F(Inf))
    -(((x - μ) / σ)^2 + F(log2π))/2 - log(σ)
end

function _loglikelihood(x::F, ::Type{<:Normal}, μ::F, σ::F) where F
    iszero(σ) && return ifelse(x == μ, F(Inf), -F(Inf))
    -(((x - μ) / σ)^2 + F(log2π))/2 - CUDAnative.log(σ)
end


# Exponential
Exponential(λ::F) where F = Exponential((λ,); val = λ * randexp(F))
Exponential{F}(λ::F) where F = Exponential(λ)

loglikelihood(x::F, ::Type{<:Exponential}, λ::F) where F =
    ifelse(x < zero(F), -F(Inf), log(λ) - λ * x)

_loglikelihood(x::F, ::Type{<:Exponential}, λ::F) where F =
    ifelse(x < zero(F), -F(Inf), CUDAnative.log(λ) - λ * x)


# Gamma
function _MarsagliaTsang2000(α::F) where F
    d = α - one(F)/3
    c = one(F) / sqrt(9d)
    while true
        x = randn(F)
        v = (one(F) + c*x)^3
        while v < zero(F)
            x = randn(F)
            v = (one(F) + c*x)^3
        end
        u = rand(F)
        # u < one(F) - F(0.0331)x^4 && return d*v
        log(u) < x^2/2 + d*(one(F) - v + log(v)) && return d*v
    end
end

function Gamma(α::F, θ::F) where F
    if α < one(F) # use the γ(1+α)*U^(1/α) trick from Marsaglia and Tsang (2000)
        x = θ * _MarsagliaTsang2000(α + one(F)) # Gamma(α + 1, θ)
        e = randexp(F)
        return Gamma((α, θ); val = x * exp(-e / α))
    elseif α == one(F)
        return Gamma((α, θ); val = θ * randexp(F))
    else
        return Gamma((α, θ); val = θ * _MarsagliaTsang2000(α))
    end
end

# TODO eliminar dependencia de StatsFuns:
loglikelihood(x::F, ::Type{<:Gamma}, α::F, θ::F) where F = gammalogpdf(α, θ, x.val)

_loglikelihood(x::F, ::Type{<:Gamma}, α::F, θ::F) where F =
    -CUDAnative.lgamma(k) - α*CUDAnative.log(θ) + (α-one(F))*CUDAnative.log(x) - x/θ
