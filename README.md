# CuNTTs

[![Build Status](https://github.com/alexp616/NTTs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexp616/NTTs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A package for computing number-theoretic transforms using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)

## Credits
**This implementation is ported to Julia from [this](https://github.com/Alisah-Ozcan/GPU-NTT) implementation.
All credit for the NTT implementations for ring sizes $2^{12}$ through $2^{28}$ go to the author, Alisah-Ozcan.**

## Usage
To start, install and import the package:

```
pkg> add NTTs

julia> using NTTs
```

Before doing anything, we need to generate an NTT plan. This step is needed
to generate roots of unity, and pre-compile the kernels with optimal
parameters. The 
line of code below generates an NTT and INTT plan using the
`plan_ntt()` function, which takes in a length $n$, a prime 
number $p$, and a primitive $n$-th root of unity of $\mathbb{F}_p$. 
A primitive $n$-th root of unity of $\mathbb{F}_p$ can be generated 
from the `primitive_nth_root_of_unity(n, p)` function. The example 
below plans an NTT with length `8`, prime `17`, and primitive $n$-th root of unity `9`.
```julia
julia> nttplan, inttplan = plan_ntt(8, 17, 9)
```
To perform the NTT, call `ntt!()` on some `CuVector` with 
the generated plan:
```julia
julia> using CUDA

julia> vec = CuArray([0, 1, 2, 3, 4, 5, 6, 7])

julia> ntt!(vec, nttplan)

julia> println(vec)
[11, 1, 12, 3, 13, 6, 14, 8]
```
To perform the INTT, you do the same thing:
```julia
julia> vec = CuArray([11, 1, 12, 3, 13, 6, 14, 8])

julia> intt!(vec, inttplan)

julia> println(vec)
[0, 1, 2, 3, 4, 5, 6, 7]
```

Note that `intt!()` automatically normalizes the output.

## Miscellaneous Notes
- The default modular reduction algorithm used is Barrett Reduction. In the future,
I may try to add options for other modular reduction algorithms, and allow users
to define their own modular reduction algorithms to pass in.
- I haven't tested anything with passing in negative integers yet, so it's up to the used to user to make sure their NTTs don't involve negative integers.
- The kernels are all optimized for `UInt64` operations, so this is the ideal type to perform NTTs with. This can be obtained by passing in `CuVector{UInt64}`'s into `ntt!()` and `intt!()`, and making sure `p` is a UInt64 in `plan_ntt()`.
- `UInt32` kernels are still faster, though not by a factor of 2. So, if a problem only requires `UInt32` precision, then it will be faster.
- See documentation of `plan_ntt()` for the `memorysafe` option, which trades off speed for holding less GPU memory.
- See documentation of `ntt!()` and `intt!()` for `bitreversedoutput` and `bitreversedinput`. If these are set to true, and the destination and source vector are the same, then the `NTT` and `INTT` are done in-place.
- The usage I developed this for takes in inputs in $\mathbb{Z}_q[x]/(x^n - 1)$, so inputs in $\mathbb{Z}_q[x]/(x^n + 1)$ are not supported. If needed, this change isn't too difficult.
