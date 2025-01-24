using NTTs
using Test
using CUDA

function test_correct()
    p = UInt32(167772161)
    for log2len in 1:4:21
        len = 1 << log2len
        npru = primitive_nth_root_of_unity(len, p)
        cpuarr = rand(zero(typeof(p)):(p - one(p)), len)
        gpuarr = CuArray(cpuarr)

        nttplan, inttplan = plan_ntt(len, p, npru)
        NTTs.cpu_ntt!(cpuarr, nttplan)
        ntt!(gpuarr, nttplan)

        @test cpuarr == Array(gpuarr)

        NTTs.cpu_intt!(cpuarr, inttplan)
        intt!(gpuarr, inttplan)

        @test cpuarr == Array(gpuarr)
    end

    p = UInt64(0x7ffffffff9000001)
    for log2len in 1:4:21
        len = 1 << log2len
        npru = primitive_nth_root_of_unity(len, p)
        cpuarr = rand(zero(typeof(p)):(p - one(p)), len)
        gpuarr = CuArray(cpuarr)

        nttplan, inttplan = plan_ntt(len, p, npru)
        NTTs.cpu_ntt!(cpuarr, nttplan)
        ntt!(gpuarr, nttplan)

        @test cpuarr == Array(gpuarr)

        NTTs.cpu_intt!(cpuarr, inttplan)
        intt!(gpuarr, inttplan)

        @test cpuarr == Array(gpuarr)
    end
end

@testset "NTTs.jl" begin
    test_correct()
end
