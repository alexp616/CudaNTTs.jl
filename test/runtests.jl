# include("../src/CudaNTTs.jl")
using CudaNTTs
using Test
using CUDA

function test_ntt()
    for pow in 1:28
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = CudaNTTs.primitive_nth_root_of_unity(n, p)
        nttplan, _ = CudaNTTs.plan_ntt(n, p, npru; memorysafe = false)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        CudaNTTs.old_ntt!(vec1, nttplan, true)
        CudaNTTs.ntt!(vec2, nttplan, true)

        @test vec1 == vec2
    end
end

function test_intt()
    for pow in 1:28
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = CudaNTTs.primitive_nth_root_of_unity(n, p)
        _, inttplan = CudaNTTs.plan_ntt(n, p, npru; memorysafe = false)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        CudaNTTs.old_intt!(vec1, inttplan, true)
        CudaNTTs.intt!(vec2, inttplan, true)

        @test vec1 == vec2
    end
end

@testset "CudaNTTs.jl" begin
    test_ntt()
    test_intt()
end
