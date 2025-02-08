include("../src/CuNTTs.jl")
using BenchmarkTools
using CUDA

function benchmark(log2n)
    n = 2^log2n
    T = UInt64
    p = T(4611685989973229569)

    npru = CuNTTs.primitive_nth_root_of_unity(n, p)
    nttplan, _ = CuNTTs.plan_ntt(n, p, npru)

    cpuvec = [T(i) for i in 1:n]

    vec2 = CuArray(cpuvec)

    CuNTTs.ntt!(vec2, nttplan, true)

    display(@benchmark CUDA.@sync CuNTTs.ntt!($vec2, $nttplan, true))
end

benchmark(24)