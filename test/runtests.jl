include("../src/NTTs.jl")
using Test
using CUDA

function test_old_ntt()
    for pow in 1:11
        n = 2 ^ pow
        T = UInt64
        p = T(65537)

        npru = NTTs.primitive_nth_root_of_unity(n, p)
        nttplan, inttplan = NTTs.plan_ntt(n, p, npru)

        cpuvec = rand(T(0):T(p - 1), n)
        cuvec = CuArray(cpuvec)

        NTTs.cpu_ntt!(cpuvec, nttplan, false)
        NTTs.old_ntt!(cuvec, nttplan, false)
        
        @test cpuvec == Array(cuvec)

        NTTs.cpu_intt!(cpuvec, inttplan, false)
        NTTs.old_intt!(cuvec, inttplan, false)

        @test cpuvec == Array(cuvec)
    end
end

function test_ntt()
    for pow in 2:28
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = NTTs.primitive_nth_root_of_unity(n, p)
        # nttplan, _ = NTTs.plan_ntt(n, p, npru)
        nttplan, _ = NTTs.plan_ntt(n, p, npru; memorysafe = true)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        NTTs.old_ntt!(vec1, nttplan, true)
        NTTs.ntt!(vec2, nttplan, true)

        @test vec1 == vec2
    end
end

function test_intt()
    for pow in 12:28
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = NTTs.primitive_nth_root_of_unity(n, p)
        # _, inttplan = NTTs.plan_ntt(n, p, npru)
        _, inttplan = NTTs.plan_ntt(n, p, npru; memorysafe = true)

        cpuvec = rand(T(0):T(p - 1), n)

        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        NTTs.old_intt!(vec1, inttplan, true)
        NTTs.intt!(vec2, inttplan, true)

        @test vec1 == vec2
    end
end


@testset "NTTs.jl" begin
    # test_old_ntt()
    test_ntt()
    test_intt()
end
