include("../src/CudaNTTs.jl")
using BenchmarkTools
using CUDA

function benchmark(log2n)
    n = 2^log2n
    T = UInt64
    p = T(4611685989973229569)

    npru = CudaNTTs.primitive_nth_root_of_unity(n, p)
    nttplan, _ = CudaNTTs.plan_ntt(n, p, npru; memoryefficient = false)

    cpuvec = [T(i) for i in 1:n]

    vec2 = CuArray(cpuvec)

    CudaNTTs.ntt!(vec2, nttplan, true)

    display(@benchmark CUDA.@sync CudaNTTs.ntt!($vec2, $nttplan, true))
end

benchmark(24)