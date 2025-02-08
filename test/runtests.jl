# include("../src/GPUNTTs.jl")
using GPUNTTs
using Test
using CUDA

function test_ntt()
    for pow in 1:28
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = GPUNTTs.primitive_nth_root_of_unity(n, p)
        nttplan, _ = GPUNTTs.plan_ntt(n, p, npru; memorysafe = false)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        GPUNTTs.old_ntt!(vec1, nttplan, true)
        GPUNTTs.ntt!(vec2, nttplan, true)

        @test vec1 == vec2
    end
end

function test_intt()
    for pow in 1:11
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = GPUNTTs.primitive_nth_root_of_unity(n, p)
        _, inttplan = GPUNTTs.plan_ntt(n, p, npru; memorysafe = false)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        GPUNTTs.old_intt!(vec1, inttplan, true)
        GPUNTTs.intt!(vec2, inttplan, true)

        @test vec1 == vec2
    end
end

@testset "GPUNTTs.jl" begin
    # test_ntt()
    test_intt()
end
