# Nondeterministics

This package is meant to provide basic infrastructure for probabilistic programming, supporting GPU calculations.
Compared to the traditional approach to statistical computing, as exemplified for instance by [Distributions.jl](https://github.com/JuliaStats/Distributions.jl),
it introduces a change of emphasis regarding random variables:

```julia
julia> x = Normal(0.,1.)
-0.3896434394237917

julia> typeof(x)
Normal{Float64}

julia> loglikelihood(x, 0., 1.)
-0.9948495381476739
```
As the name suggests, this code is nondeterministic:
```julia
 julia> Normal(0.,1.)
0.11046163865819952
```
Thus, `Normal` here is like an elementary [Gen.jl](https://probcomp.github.io/Gen/) generating function. Mathematical operations can be performed on nondeterministics, but the type of the result ceases to carry information on its distribution:
```julia
julia> x = Normal(0.,1.) + Gamma(.5,.5)
0.07986315668099568

julia> typeof(x)
Float64
```
More generally, nondeterministics behave as regular values:
```julia
julia> xs = [Uniform(0.,1.) for _ in 1:100]
100-element Array{Uniform{Float64},1}:
 0.222925644855418   
 0.8334598869710552  
 0.2587837916093274  
 0.5241714629025322  
 0.42892008394539904
 0.022817128564959432
 0.7752682852175297  
 0.5866079287907486  
 0.7589699187139316  
 0.17452322903655682
 0.6412426210242628  
 0.24467964148191634
 0.5715608429738686  
 ⋮                   
 0.5315595170455634  
 0.30358985676838035
 0.8371478661168452  
 0.7194761175815878  
 0.16874598895687165
 0.025319320917682298
 0.7021787389536511  
 0.810540331217952   
 0.4019100617968885  
 0.776886486329488   
 0.3357906785509175  
 0.42052383699225726
 0.8447131631997253  

julia> extrema(xs)
(0.005848065274397518, 0.9968539798022258)

julia> sum(xs) / length(xs)
0.5198061872434725

julia> p = Dirichlet{100}(.5)
100-element Dirichlet{100,Float64}:
 0.001241478367202549  
 3.845450568319268e-5  
 0.013600473382295022  
 0.08667284425068368   
 0.004177429408020993  
 0.004680152031659506  
 0.0011456114625197796
 0.012364974229756687  
 0.00016757871413004109
 0.0028056728084065756
 0.022787785473215992  
 0.041210742672776766  
 0.0012523253622531499
 ⋮                     
 0.0024566945833043026
 0.0026977413604141708
 0.008790320092977028  
 0.010420062851195502  
 0.005585138876396112  
 0.0006717313600257085
 3.600723856318514e-5  
 0.0018162426617226778
 0.0012115953364742753
 0.017286632980644992  
 0.001509384288078791  
 0.0006308184486443109
 0.010372540191744165  

julia> i = Categorical(p)
57

julia> xs[i]
0.15478769930723413
```


## Some GPU examples

Generate a 3x4 CuArray of normally distributed random variables, with random means and unit variance, and compute its loglikelihood assuming it is an iid array $\sim Normal(0,1)$:

```julia
julia> p = Product{Normal}(CuArrays.rand(3,4), CuArray([1f0]))
3×4 Product{Normal{Float32},2,CuArray{Normal{Float32},2,Nothing}}:
 0.675282  0.840436  -0.0798901  -0.623571
 0.102764  1.20615   -1.26977     0.423147
 1.4656    1.27592    1.46537     0.837499

julia> loglikelihood(p, CuArray([0f0]), CuArray([1f0]))
-16.746746f0
```

Now, generate a normally distributed random variable and compute in parallel its likelihood with respect to several different mean parameters:
```julia
julia> x = Normal(0.,1.)
-0.7584109786995538

julia> loglikelihood.(CuArray([x]), CuArray(collect(-10.:1.:10.)), CuArray([1.]))
21-element CuArray{Float64,1,Nothing}:
 -43.62242235251515  
 -34.8808333312147   
 -27.13924430991425  
 -20.3976552886138   
 -14.656066267313356
  -9.914477246012911
  -6.172888224712465
  -3.431299203412019
  -1.6897101821115725
  -0.9481211608111265
  -1.2065321395106803
  -2.4649431182102344
  -4.723354096909788
  -7.981765075609342
 -12.240176054308895
 -17.498587033008448
 -23.756998011708003
 -31.015408990407558
 -39.2738199691071   
 -48.53223094780666  
 -58.79064192650621  
```
