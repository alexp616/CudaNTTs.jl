include("../src/NTTs.jl")
using Test
using CUDA
using BenchmarkTools

function test_correct()
    n = 2^25
    T = UInt64
    p = T(4611685989973229569)
    # n = 2^10
    # T = UInt64
    # p = T(65537)
    npru = NTTs.primitive_nth_root_of_unity(n, p)
    # println("npru: $npru")
    plan = NTTs.plan_ntt(n, p, npru)

    cpuvec = rand(T(0):T(p - 1), n)
    # cpuvec = [T(i) for i in 1:n]
    # cpuvec = ones(T, n)

    vec1 = CuArray(cpuvec)
    vec2 = CuArray(cpuvec)

    # NTTs.cpu_ntt!(cpuvec, plan)
    NTTs.old_ntt!(vec1, plan)
    NTTs.ntt!(vec2, plan)

    # @assert cpuvec == Array(vec1)
    @assert vec1 == vec2
end

function benchmark()
    n = 2^28
    T = UInt64
    p = T(4611685989973229569)

    npru = NTTs.primitive_nth_root_of_unity(n, p)
    plan = NTTs.plan_ntt(n, p, npru)

    cpuvec = [T(i) for i in 1:n]

    vec2 = CuArray(cpuvec)

    NTTs.ntt!(vec2, plan)

    display(@benchmark CUDA.@sync NTTs.ntt!($vec2, $plan, true))
    # for i in 1:100
    #     CUDA.@time NTTs.ntt!(vec2, plan, true)
    # end

    # CUDA.@profile NTTs.ntt!(vec, plan, true)
end

@testset "NTTs.jl" begin
    # test_correct()
    benchmark()
end
