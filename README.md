# NTTs

[![Build Status](https://github.com/alexp616/NTTs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexp616/NTTs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A package for computing number-theoretic transforms using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)

This implementation is ported to Julia from [this](https://github.com/Alisah-Ozcan/GPU-NTT) implementation.
All credit goes to the authors of that repo.

## Usage
To start, install and import the package:

```
pkg> add https://github.com/alexp616/NTTs.jl

julia> using NTTs
```

Before doing anything, you need to generate an NTT plan. The 
line of code below generates an NTT and INTT plan using the
`plan_ntt()` function, which takes in a length $n$, a prime 
number $p$, and a primitive $n$-th root of unity of $\mathbb{F}_p$. 
A primitive $n$-th root of unity of $\mathbb{F}_p$ can be generated 
from the `primitive_nth_root_of_unity(n, p)` function. The example 
below plans an NTT with length `8`, prime `17`, and primitive $n$-th root of unity `9`.
```
julia> nttplan, inttplan = plan_ntt(8, 17, 9)
```
To perform the NTT, you call `ntt!()` on some `CuVector` with 
the generated plan:
```
julia> using CUDA

julia> vec = CuArray([0, 1, 2, 3, 4, 5, 6, 7])

julia> ntt!(vec, nttplan)

julia> println(vec)
[11, 1, 12, 3, 13, 6, 14, 8]
```
To perform the INTT, you do the same thing:
```
julia> vec = CuArray([11, 1, 12, 3, 13, 6, 14, 8])

julia> intt!(vec, inttplan)

julia> println(vec)
[0, 1, 2, 3, 4, 5, 6, 7]
```

Note that `intt!()` automatically normalizes the output, 
and both `ntt!()` and `intt!()` change the entries of the 
input vector without returning anything.

## Future Work
In the future, I may make the package work under AbstractFFTs, but
for my current use case, I just have this.

