include("../src/NTTs.jl")
using Test
using CUDA
using BenchmarkTools

function test_correct()

    for pow in 12:25
        n = 2 ^ pow
        T = UInt64
        p = T(4611685989973229569)

        npru = NTTs.primitive_nth_root_of_unity(n, p)
        nttplan, inttplan = NTTs.plan_ntt(n, p, npru; memorysafe = true)

        cpuvec = rand(T(0):T(p - 1), n)


        vec1 = CuArray(cpuvec)
        vec2 = CuArray(cpuvec)

        # NTTs.cpu_ntt!(cpuvec, plan)
        NTTs.old_ntt!(vec1, nttplan)
        NTTs.ntt!(vec2, nttplan)

        # @assert cpuvec == Array(vec1)
        @test vec1 == vec2
    end
end

function benchmark()
    n = 2^20
    T = UInt64
    p = T(4611685989973229569)

    npru = NTTs.primitive_nth_root_of_unity(n, p)
    nttplan, inttplan = NTTs.plan_ntt(n, p, npru)

    cpuvec = [T(i) for i in 1:n]

    vec2 = CuArray(cpuvec)

    NTTs.ntt!(vec2, nttplan, inttplan)

    display(@benchmark CUDA.@sync NTTs.ntt!($vec2, $nttplan, inttplan, true))
    # for i in 1:100
    #     CUDA.@time NTTs.ntt!(vec2, plan, true)
    # end

    # CUDA.@profile NTTs.ntt!(vec, plan, true)
end

@testset "NTTs.jl" begin
    test_correct()
    # benchmark()
end
