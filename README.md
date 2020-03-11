# Nondeterministics

This package is meant to provide basic infrastructure for probabilistic programming, with CuArrays support and Tracker/Zygote compatibility.
Compared to [Distributions.jl](https://github.com/JuliaStats/Distributions.jl),
it introduces a change of emphasis regarding random variables:

```julia
julia> x = Normal(0,1)
0.1487326095231564

julia> logpdf(Normal)(x, 0, 1)
-0.9299992277724566

julia> Categorical(Dirichlet(1 + Poisson(10), Gamma(.5, .5))...)
2
```
As the name suggests, this code is nondeterministic.
Thus, distributions here are like elementary [Gen.jl](https://probcomp.github.io/Gen/) generating functions.

## Some GPU examples

Generate a 3x4 CuArray of normally distributed random variables, with random means and unit variance, and compute its loglikelihood assuming it is an iid array $\sim Normal(0,1)$:

```julia
julia> p = CuArray(Normal.(rand(3,4), 1.))
3×4 CuArray{Float64,2,Nothing}:
  1.61652   1.19637   0.87373   -1.66683
  1.48393  -0.335277  0.456668  -0.073796
 -1.3067    1.87748   0.68738    0.692792

julia> sum(logpdf(Normal).(p, 0., 1.))
-19.177009448710894
```

Now, generate a normally distributed random variable and compute in parallel its likelihood with respect to several different mean parameters:
```julia
julia> x = Normal(0.,1.)
-0.21472810689915206

julia> logpdf(Normal).(x, CuArray(collect(-10.:1.:10.)), 1.)
21-element CuArray{Float64,1,Nothing}:
 -48.794711544159405
 -39.50943965105856  
 -31.224167757957705
 -23.938895864856857
 -17.65362397175601  
 -12.36835207865516  
  -8.08308018555431  
  -4.797808292453463
  -2.5125363993526157
  -1.2272645062517675
  -0.9419926131509196
  -1.6567207200500715
  -3.371448826949224
  -6.086176933848376
  -9.800905040747526
 -14.515633147646678
 -20.23036125454583  
 -26.945089361444982
 -34.65981746834414  
 -43.37454557524329  
 -53.08927368214244  
```

## A differentiable programming example

The following code makes a simple maximum likelihood estimation using Zygote.
```julia
Zygote.@adjoint logpdf(d) = logpdf(d), _ -> 0

n = 10000
data = CuArray([Gamma(.5,.9) for _ in 1:n]);
μ, σ = Uniform(0,5), Uniform(0,5);

ϵ = .01
for i in 1:n
    dμ, dσ = gradient(μ, σ) do μ, σ
        sum(logpdf(Gamma).(data, μ, σ)) / n
    end
    μ += ϵ * dμ; σ += ϵ * dσ
    i % 1000 == 0 && @show μ, σ
end
```
