CC89() = string(capability(device())) == "8.9.0"

"""
    struct NTTPlan{T<:Union{Int32, Int64, UInt32, UInt64}}

Struct containing all information necessary to perform a NTT.
Either construct directly through `NTTPlan(n, p, npru)`, or see
`plan_ntt()`
"""
struct NTTPlan{T<:INTTYPES}
    n::Int32
    p::T
    reducer::Reducer{T}
    npru::T
    log2len::Int32
    rootOfUnityTable::Union{CuVector{T}, T}
    compiledKernels::Vector{Function}

    function NTTPlan(n::Integer, p::T, npru::T; memoryefficient = false) where T<:Integer
        @assert ispow2(n)
        @assert p % n == 1
        n = Int32(n)
        log2n = intlog2(n)

        reducer = BarrettReducer(p)
        if memoryefficient
            rootOfUnityTable = npru
        else
            rootOfUnityTable = gpu_root_of_unity_table_generator(npru, reducer, n ÷ 2)
        end

        if log2n <= 11
            compiledKernels = Function[]
            temp = CUDA.zeros(T, 1)

            if memoryefficient
                kernel = @cuda launch=false me_small_ntt_kernel!(temp, temp, rootOfUnityTable, reducer, log2n)
            else
                kernel = @cuda launch=false small_ntt_kernel!(temp, temp, rootOfUnityTable, reducer, log2n)
            end
            func(in, out) = kernel(in, out, rootOfUnityTable, reducer, log2n; threads = n ÷ 2, blocks = 1, shmem = sizeof(T) * n)
            push!(compiledKernels, func)
            return new{T}(n, p, reducer, npru, log2n, rootOfUnityTable, compiledKernels)
        elseif log2n <= 28
            cfgs = KernelConfig[]
            if log2n == 12
                push!(cfgs, KernelConfig(8, 1, 64, 4, 512 * sizeof(T), 8, 0, 0, 3, true))
                push!(cfgs, KernelConfig(1, 8, 256, 1, 512 * sizeof(T), 8, 3, 0, 9, false))
            elseif log2n == 13
                push!(cfgs, KernelConfig(16, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true))
                push!(cfgs, KernelConfig(1, 16, 256, 1, 512 * sizeof(T), 8, 4, 0, 9, false))
            elseif log2n == 14
                push!(cfgs, KernelConfig(32, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true))
                push!(cfgs, KernelConfig(1, 32, 256, 1, 512 * sizeof(T), 8, 5, 0, 9, false))
            elseif log2n == 15
                push!(cfgs, KernelConfig(64, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true))
                push!(cfgs, KernelConfig(1, 64, 256, 1, 512 * sizeof(T), 8, 6, 0, 9, false))
            elseif log2n == 16
                push!(cfgs, KernelConfig(128, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true))
                push!(cfgs, KernelConfig(1, 128, 256, 1, 512 * sizeof(T), 8, 7, 0, 9, false))
            elseif log2n == 17
                push!(cfgs, KernelConfig(256, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true))
                push!(cfgs, KernelConfig(16, 16, 32, 8, 512 * sizeof(T), 8, 4, 0, 4, true))
                push!(cfgs, KernelConfig(1, 256, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, false))
            elseif log2n == 18
                push!(cfgs, KernelConfig(512, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true))
                push!(cfgs, KernelConfig(32, 16, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true))
                push!(cfgs, KernelConfig(1, 512, 256, 1, 512 * sizeof(T), 8, 9, 0, 9, false))
            elseif log2n == 19
                push!(cfgs, KernelConfig(1024, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true))
                push!(cfgs, KernelConfig(32, 32, 16, 16, 512 * sizeof(T), 8, 5, 0, 5, true))
                push!(cfgs, KernelConfig(1, 1024, 256, 1, 512 * sizeof(T), 8, 10, 0, 9, false))
            elseif log2n == 20
                push!(cfgs, KernelConfig(2048, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true))
                push!(cfgs, KernelConfig(64, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
                push!(cfgs, KernelConfig(1, 2048, 256, 1, 512 * sizeof(T), 8, 11, 0, 9, false))
            elseif log2n == 21
                push!(cfgs, KernelConfig(4096, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true))
                push!(cfgs, KernelConfig(64, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true))
                push!(cfgs, KernelConfig(1, 4096, 256, 1, 512 * sizeof(T), 8, 12, 0, 9, false))
            elseif log2n == 22
                push!(cfgs, KernelConfig(8192, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true))
                push!(cfgs, KernelConfig(128, 64, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true))
                push!(cfgs, KernelConfig(1, 8192, 256, 1, 512 * sizeof(T), 8, 13, 0, 9, false))
            elseif log2n == 23
                push!(cfgs, KernelConfig(16384, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true))
                push!(cfgs, KernelConfig(128, 128, 4, 64, 512 * sizeof(T), 8, 7, 0, 7, true))
                push!(cfgs, KernelConfig(1, 16384, 256, 1, 512 * sizeof(T), 8, 14, 0, 9, false))
            elseif log2n == 24
                push!(cfgs, KernelConfig(16384, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true))
                push!(cfgs, KernelConfig(128, 128, 8, 64, 1024 * sizeof(T), 9, 7, 0, 7, true))
                push!(cfgs, KernelConfig(1, 16384, 512, 1, 1024 * sizeof(T), 9, 14, 0, 10, false))
            elseif log2n == 25
                push!(cfgs, KernelConfig(32768, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true))
                push!(cfgs, KernelConfig(256, 128, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true))
                push!(cfgs, KernelConfig(32768, 1, 512, 1, 1024 * sizeof(T), 9, 15, 0, 10, false))
            elseif log2n == 26
                push!(cfgs, KernelConfig(65536, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true))
                push!(cfgs, KernelConfig(256, 256, 4, 128, 1024 * sizeof(T), 9, 8, 0, 8, true))
                push!(cfgs, KernelConfig(65536, 1, 512, 1, 1024 * sizeof(T), 9, 16, 0, 10, false))
            elseif log2n == 27 && CC89()
                push!(cfgs, KernelConfig(131072, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true))
                push!(cfgs, KernelConfig(512, 256, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true))
                push!(cfgs, KernelConfig(131072, 1, 512, 1, 1024 * sizeof(T), 9, 17, 0, 10, false))
            elseif log2n == 28 && CC89()
                push!(cfgs, KernelConfig(262144, 1, 2, 256, 1024 * sizeof(T), 9, 0, 0, 9, true))
                push!(cfgs, KernelConfig(512, 512, 2, 256, 1024 * sizeof(T), 9, 9, 0, 9, true))
                push!(cfgs, KernelConfig(262144, 1, 512, 1, 1024 * sizeof(T), 9, 18, 0, 10, false))
            elseif log2n == 27
                push!(cfgs, KernelConfig(262144, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true))
                push!(cfgs, KernelConfig(8192, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
                push!(cfgs, KernelConfig(128, 2048, 4, 64, 512 * sizeof(T), 8, 11, 0, 7, true))
                push!(cfgs, KernelConfig(262144, 1, 256, 1, 512 * sizeof(T), 8, 18, 0, 9, false))
            elseif log2n == 28
                push!(cfgs, KernelConfig(524288, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true)),
                push!(cfgs, KernelConfig(8192, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true))
                push!(cfgs, KernelConfig(128, 4096, 4, 64, 512 * sizeof(T), 8, 12, 0, 7, true))
                push!(cfgs, KernelConfig(524288, 1, 256, 1, 512 * sizeof(T), 8, 19, 0, 9, false))
            end

            if log2n < 25
                compiledKernels = map(params -> compile_kernel(params, log2n, reducer, rootOfUnityTable), cfgs)
            else
                compiledKernels = Function[]
                for i in 1:length(cfgs) - 1
                    push!(compiledKernels, compile_kernel(cfgs[i], log2n, reducer, rootOfUnityTable))
                end
                push!(compiledKernels, compile_kernel(cfgs[end], log2n, reducer, rootOfUnityTable, false))
            end
            
            return new{T}(n, p, reducer, npru, log2n, rootOfUnityTable, compiledKernels)
        else
            throw("Ring size not supported yet.")
        end
    end
end

"""
    struct INTTPlan{T<:Union{Int32, Int64, UInt32, UInt64}}

Struct containing all information necessary to perform a NTT.
Either construct directly through `INTTPlan(n, p, npru)`, or see
`plan_ntt()`
"""
struct INTTPlan{T<:INTTYPES}
    n::Int32
    p::T
    reducer::Reducer{T}
    npruinv::T
    n_inverse::T
    log2len::Int32
    rootOfUnityTable::Union{CuVector{T}, T}
    compiledKernels::Vector{Function}

    function INTTPlan(n::Integer, p::T, npru::T; memoryefficient = false) where T<:INTTYPES
        @assert ispow2(n)
        @assert p % n == 1
        n = Int32(n)
        log2n = intlog2(n)

        npruinv = T(invmod(BigInt(npru), BigInt(p)))
        @assert BigInt(npruinv) * BigInt(npru) % p == 1 # because I don't trust invmod
        n_inverse = T(invmod(BigInt(n), BigInt(p)))
        @assert BigInt(n) * BigInt(n_inverse) % p == 1

        reducer = BarrettReducer(p)
        if memoryefficient
            rootOfUnityTable = npruinv
        else
            rootOfUnityTable = gpu_root_of_unity_table_generator(npruinv, reducer, n ÷ 2)
        end

        if log2n <= 11
            compiledKernels = Function[]
            temp = CUDA.zeros(T, 1)

            if memoryefficient
                kernel = @cuda launch=false me_small_intt_kernel!(temp, temp, rootOfUnityTable, reducer, log2n, n_inverse)
            else
                kernel = @cuda launch=false small_intt_kernel!(temp, temp, rootOfUnityTable, reducer, log2n, n_inverse)
            end
            func(in, out) = kernel(in, out, rootOfUnityTable, reducer, log2n, n_inverse; threads = n ÷ 2, blocks = 1, shmem = sizeof(T) * n)
            push!(compiledKernels, func)

            return new{T}(n, p, reducer, npruinv, n_inverse, log2n, rootOfUnityTable, compiledKernels)
        elseif log2n <= 28
            cfgs = KernelConfig[]
            if log2n == 12
                push!(cfgs, KernelConfig(1, 8, 256, 1, 512 * sizeof(T), 8, 11, 3, 9, false))
                push!(cfgs, KernelConfig(8, 1, 64, 4, 512 * sizeof(T), 8, 2, 0, 3, true))
            elseif log2n == 13
                push!(cfgs, KernelConfig(1, 16, 256, 1, 512 * sizeof(T), 8, 12, 4, 9, false))
                push!(cfgs, KernelConfig(16, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true))
            elseif log2n == 14
                push!(cfgs, KernelConfig(1, 32, 256, 1, 512 * sizeof(T), 8, 13, 5, 9, false))
                push!(cfgs, KernelConfig(32, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true))
            elseif log2n == 15
                push!(cfgs, KernelConfig(1, 64, 256, 1, 512 * sizeof(T), 8, 14, 6, 9, false))
                push!(cfgs, KernelConfig(64, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
            elseif log2n == 16
                push!(cfgs, KernelConfig(1, 128, 256, 1, 512 * sizeof(T), 8, 15, 7, 9, false))
                push!(cfgs, KernelConfig(128, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true))
            elseif log2n == 17
                push!(cfgs, KernelConfig(1, 256, 256, 1, 512 * sizeof(T), 8, 16, 8, 9, false))
                push!(cfgs, KernelConfig(16, 16, 32, 8, 512 * sizeof(T), 8, 7, 4, 4, false))
                push!(cfgs, KernelConfig(256, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true))
            elseif log2n == 18
                push!(cfgs, KernelConfig(1, 512, 256, 1, 512 * sizeof(T), 8, 17, 9, 9, false))
                push!(cfgs, KernelConfig(32, 16, 16, 16, 512 * sizeof(T), 8, 8, 4, 5, false))
                push!(cfgs, KernelConfig(512, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true))
            elseif log2n == 19
                push!(cfgs, KernelConfig(1, 1024, 256, 1, 512 * sizeof(T), 8, 18, 10, 9, false))
                push!(cfgs, KernelConfig(32, 32, 16, 16, 512 * sizeof(T), 8, 9, 5, 5, false))
                push!(cfgs, KernelConfig(1024, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true))
            elseif log2n == 20
                push!(cfgs, KernelConfig(1, 2048, 256, 1, 512 * sizeof(T), 8, 19, 11, 9, false))
                push!(cfgs, KernelConfig(64, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false))
                push!(cfgs, KernelConfig(2048, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true))
            elseif log2n == 21
                push!(cfgs, KernelConfig(1, 4096, 256, 1, 512 * sizeof(T), 8, 20, 12, 9, false))
                push!(cfgs, KernelConfig(64, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false))
                push!(cfgs, KernelConfig(4096, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
            elseif log2n == 22
                push!(cfgs, KernelConfig(1, 8192, 256, 1, 512 * sizeof(T), 8, 21, 13, 9, false))
                push!(cfgs, KernelConfig(128, 64, 4, 64, 512 * sizeof(T), 8, 12, 6, 7, false))
                push!(cfgs, KernelConfig(8192, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
            elseif log2n == 23
                push!(cfgs, KernelConfig(1, 16384, 256, 1, 512 * sizeof(T), 8, 22, 14, 9, false))
                push!(cfgs, KernelConfig(128, 128, 4, 64, 512 * sizeof(T), 8, 13, 7, 7, false))
                push!(cfgs, KernelConfig(16384, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true))
            elseif log2n == 24
                push!(cfgs, KernelConfig(1, 16384, 512, 1, 1024 * sizeof(T), 9, 23, 14, 10, false))
                push!(cfgs, KernelConfig(128, 128, 8, 64, 1024 * sizeof(T), 9, 13, 7, 7, false))
                push!(cfgs, KernelConfig(16384, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true))
            elseif log2n == 25
                push!(cfgs, KernelConfig(32768, 1, 512, 1, 1024 * sizeof(T), 9, 24, 15, 10, false))
                push!(cfgs, KernelConfig(256, 128, 4, 128, 1024 * sizeof(T), 9, 14, 7, 8, false))
                push!(cfgs, KernelConfig(32768, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true))
            elseif log2n == 26
                push!(cfgs, KernelConfig(65536, 1, 512, 1, 1024 * sizeof(T), 9, 25, 16, 10, false))
                push!(cfgs, KernelConfig(256, 256, 4, 128, 1024 * sizeof(T), 9, 15, 8, 8, false))
                push!(cfgs, KernelConfig(65536, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true))
            elseif log2n == 27 && CC89()
                push!(cfgs, KernelConfig(131072, 1, 512, 1, 1024 * sizeof(T), 9, 26, 17, 10, false))
                push!(cfgs, KernelConfig(512, 256, 2, 256, 1024 * sizeof(T), 9, 16, 8, 9, false))
                push!(cfgs, KernelConfig(131072, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true))
            elseif log2n == 28 && CC89()
                push!(cfgs, KernelConfig(262144, 1, 512, 1, 1024 * sizeof(T), 9, 27, 18, 10, false))
                push!(cfgs, KernelConfig(512, 512, 2, 256, 1024 * sizeof(T), 9, 17, 9, 9, false))
                push!(cfgs, KernelConfig(262144, 1, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true))
            elseif log2n == 27
                push!(cfgs, KernelConfig(262144, 1, 256, 1, 512 * sizeof(T), 8, 26, 18, 9, false))
                push!(cfgs, KernelConfig(128, 2048, 4, 64, 512 * sizeof(T), 8, 17, 11, 7, false))
                push!(cfgs, KernelConfig(8192, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false))
                push!(cfgs, KernelConfig(262144, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true))
            elseif log2n == 28
                push!(cfgs, KernelConfig(524288, 1, 256, 1, 512 * sizeof(T), 8, 27, 19, 9, false))
                push!(cfgs, KernelConfig(128, 4096, 4, 64, 512 * sizeof(T), 8, 18, 12, 7, false))
                push!(cfgs, KernelConfig(8192, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false))
                push!(cfgs, KernelConfig(524288, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true))
            end

            if log2n < 25
                compiledKernels = map(params -> compile_kernel(params, n_inverse, log2n, reducer, rootOfUnityTable), cfgs)
            else
                compiledKernels = Function[]

                push!(compiledKernels, compile_kernel(cfgs[1], n_inverse, log2n, reducer, rootOfUnityTable, false))
                for i in 2:length(cfgs)
                    push!(compiledKernels, compile_kernel(cfgs[i], n_inverse, log2n, reducer, rootOfUnityTable))
                end
            end

            return new{T}(n, p, reducer, npruinv, n_inverse, log2n, rootOfUnityTable, compiledKernels)
        else
            throw("Ring size not supported")
        end
    end
end

struct KernelConfig
    griddim_x::Int
    griddim_y::Int
    blockdim_x::Int
    blockdim_y::Int
    shared_memory::Int

    shared_index::Int32
    logm::Int32
    k::Int32
    outer_iteration_count::Int32

    not_last_kernel::Bool

    function KernelConfig(griddim_x::Int, griddim_y::Int, blockdim_x::Int, blockdim_y::Int, shared_memory::Int, shared_index::Int, logm::Int, k::Int, outer_iteration_count::Int, not_last_kernel::Bool)
        return new(griddim_x, griddim_y, blockdim_x, blockdim_y, shared_memory, Int32(shared_index), Int32(logm), Int32(k), Int32(outer_iteration_count), not_last_kernel)
    end
end

function compile_kernel(params::KernelConfig, log2n::Int32, modulus::Reducer{T}, rouTable::Union{CuVector{T}, T}, standard::Bool = true) where T<:INTTYPES
    temp = CUDA.zeros(T, 1)
    shmem_length = Int32(params.shared_memory ÷ sizeof(T))

    # @device_code_ptx kernel = @cuda launch=false ntt_kernel1!(temp, temp, temp, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
    # println(occupancy(kernel.fun, 1024; shmem = 1024 * 2 * sizeof(T)))
    # throw("sigma")

    if rouTable isa CuVector{T}
        if standard
            kernel = @cuda launch=false ntt_kernel1!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
        else
            kernel = @cuda launch=false ntt_kernel2!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
        end
    else
        # rouTable isn't actually a table, but just the root of unity
        if standard
            kernel = @cuda launch=false me_ntt_kernel1!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
        else
            kernel = @cuda launch=false me_ntt_kernel2!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
        end
    end

    func(in, out) = kernel(in, out, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel; threads = (params.blockdim_x, params.blockdim_y), blocks = (params.griddim_x, params.griddim_y), shmem = params.shared_memory)

    return func
end

function compile_kernel(params::KernelConfig, n_inverse::T, log2n::Int32, modulus::Reducer{T}, rouTable::Union{CuVector{T}, T}, standard::Bool = true) where T<:INTTYPES
    temp = CUDA.zeros(T, 1)
    shmem_length = Int32(params.shared_memory ÷ sizeof(T))

    if rouTable isa CuVector{T}
        if standard
            kernel = @cuda launch=false intt_kernel1!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.k, params.outer_iteration_count, log2n, shmem_length, n_inverse, params.not_last_kernel)
        else
            kernel = @cuda launch=false intt_kernel2!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.k, params.outer_iteration_count, log2n, shmem_length, n_inverse, params.not_last_kernel)
        end
    else
        # rouTable isn't actually a table, but just the root of unity
        if standard
            kernel = @cuda launch=false me_intt_kernel1!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.k, params.outer_iteration_count, log2n, shmem_length, n_inverse, params.not_last_kernel)
        else
            kernel = @cuda launch=false me_intt_kernel2!(temp, temp, rouTable, modulus, params.shared_index, params.logm, params.k, params.outer_iteration_count, log2n, shmem_length, n_inverse, params.not_last_kernel)
        end
    end

    func(in, out) = kernel(in, out, rouTable, modulus, params.shared_index, params.logm, params.k, params.outer_iteration_count, log2n, shmem_length, n_inverse, params.not_last_kernel; threads = (params.blockdim_x, params.blockdim_y), blocks = (params.griddim_x, params.griddim_y), shmem = params.shared_memory)

    return func
end

"""
    plan_ntt(len::Integer, p::Integer, npru::Integer; memoryefficient = false) -> Tuple{NTTPlan, INTTPlan}

Returns a NTTPlan, as well as the inverse INTTPlan to be used in 
`ntt!()` and `intt!()`. Type of NTT is determined by p, which must be in
`Union{Int32, Int64, UInt32, UInt64}`.

# Arguments:
- `len`: Length of NTT (must be power of 2)
- `p`: Characteristic of field to perform NTT in.
- `npru`: len-th primitive root of unity of `p`. No validation is done, see `primitive_nth_root_of_unity()` to generate.
- `memoryefficient`: Boolean to determine whether or not to generate root of unity table. If false, NTT will be slower but use half the memory.
"""
function plan_ntt(len::Integer, p::INTTYPES, npru::INTTYPES; memoryefficient = false)::Tuple{NTTPlan, INTTPlan}
    @assert ispow2(len) "len must be a power of 2."
    @assert isprime(p) "p must be prime."
    @assert p < typemax(typeof(p)) >> 2 "p must be smaller than typemax(p) for Barrett reduction"
    npru = typeof(p)(npru)
    # @assert is_primitive_root(npru, p, n) this computation takes too long

    return NTTPlan(len, p, npru; memoryefficient = memoryefficient), INTTPlan(len, p, npru; memoryefficient = memoryefficient)
end

"""
    ntt!(vec::CuVector{T}, dest::CuVector{T}, plan::NTTPlan{T}, bitreversedoutput::Bool = false) where T<:Union{Int32, Int64, UInt32, UInt64}

Takes the NTT according to `plan` of `vec`, storing the result in `dest`. 

# Arguments:
- `vec`: Source vector of NTT.
- `dest`: Destination vector of NTT.
- `plan`: NTTPlan with ntt information. See `plan_ntt()` for generating plans.
- `bitreversedoutput`: Bool determining whether output is in bit-reversed order. Defaults to false.

If `vec` and `dest` are the same, and `bitreversedoutput` is true, then
the NTT is done in-place. As a shortcut, an overload is provided:

```jldoctest
ntt!(vec::CuVector, plan::NTTPlan, bitreversedoutput::Bool = false)
```
"""
function ntt!(vec::CuVector{T}, dest::CuVector{T}, plan::NTTPlan{T}, bitreversedoutput::Bool = false) where T<:INTTYPES
    @assert intlog2(length(vec)) == plan.log2len

    plan.compiledKernels[1](vec, dest)
    for i in 2:length(plan.compiledKernels)
        plan.compiledKernels[i](dest, dest)
    end

    if !bitreversedoutput
        correct = parallel_bit_reverse_copy(dest)
        dest .= correct
    end

    return nothing
end

function ntt!(vec::CuVector{T}, plan::NTTPlan{T}, bitreversedoutput::Bool = false) where T<:INTTYPES
    return ntt!(vec, vec, plan, bitreversedoutput)
end

"""
    intt!(vec::CuVector{T}, dest::CuVector{T}, plan::INTTPlan{T}, bitreversedinput::Bool = false) where T<:Union{Int32, Int64, UInt32, UInt64}

Takes the INTT according to `plan` of `vec`, storing the result in `dest`. 

# Arguments:
- `vec`: Source vector of INTT.
- `dest`: Destination vector of INTT.
- `plan`: INTTPlan with intt information. See `plan_ntt()` for generating plans.
- `bitreversedinput`: Bool determining whether input is in bit-reversed order. Defaults to false.

If `vec` and `dest` are the same, and `bitreversedoutput` is true, then
the INTT is done in-place. As a shortcut, an overload is provided:

```jldoctest
intt!(vec::CuVector, plan::INTTPlan, bitreversedinput::Bool = false)
```
"""
function intt!(vec::CuVector{T}, dest::CuVector{T}, plan::INTTPlan{T}, bitreversedinput::Bool = false) where T<:INTTYPES
    @assert intlog2(length(vec)) == plan.log2len

    if !bitreversedinput
        correct = parallel_bit_reverse_copy(vec)
        vec .= correct
    end

    plan.compiledKernels[1](vec, dest)
    for i in 2:length(plan.compiledKernels)
        plan.compiledKernels[i](dest, dest)
    end

    return nothing
end

function intt!(vec::CuVector{T}, plan::INTTPlan{T}, bitreversedinput::Bool = false) where T<:INTTYPES
    return intt!(vec, vec, plan, bitreversedinput)
end