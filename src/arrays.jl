function randDirichlet(α::T...) where T<:Real
    p = SVector((αi -> Gamma(αi,one(T))).(α))
    p ./ sum(p)
end
randDirichlet(α::Integer...) = randDirichlet(Float64.(α)...)
randDirichlet(n::Integer, α::Real) = randDirichlet((α for _ in 1:n)...)
randDirichlet(n::Integer, α::Integer) = randDirichlet(n, Float64(α))

function logpdfDirichlet(x, α::Real...)
    a, b = sum(u -> SVector(u,loggamma(u)), α)
    s = sum(((u,v) -> (u-one(u))log(v)).(α, x))
    s - b + loggamma(a)
end

function _logpdfDirichlet(x, α::Real...)
    a, b = sum(u -> SVector(u,CUDAnative.lgamma(u)), α)
    s = sum(((u,v) -> (u-one(u))CUDAnative.log(v)).(α, x))
    s - b + CUDAnative.lgamma(a)
end
