

struct NTTPlan{T<:Unsigned}
    n::Int32
    p::T
    reducer::Reducer{T}
    npru::T
    log2len::Int32
    rootOfUnityTable::Union{CuVector{T}, Vector{T}}
    compiledKernels::Vector{Function}

    function NTTPlan(n::Integer, p::T, npru::T; memorysafe = false) where T<:Integer
        @assert ispow2(n)
        @assert p % (2 * n) == 1
        n = Int32(n)
        log2n = intlog2(n)

        reducer = BarrettReducer(p)
        if memorysafe
            rootOfUnityTable = root_of_unity_table_generator(modsqrt(npru, p), reducer, n)
        else
            rootOfUnityTable = gpu_root_of_unity_table_generator(modsqrt(npru, p), reducer, n)
        end

        if log2n <= 11
            return new{T}(n, p, reducer, npru, log2n, rootOfUnityTable, Function[])
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
            elseif log2n == 27
                push!(cfgs, KernelConfig(131072, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true))
                push!(cfgs, KernelConfig(512, 256, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true))
                push!(cfgs, KernelConfig(131072, 1, 512, 1, 1024 * sizeof(T), 9, 17, 0, 10, false))
            elseif log2n == 28
                push!(cfgs, KernelConfig(262144, 1, 2, 256, 1024 * sizeof(T), 9, 0, 0, 9, true))
                push!(cfgs, KernelConfig(512, 512, 2, 256, 1024 * sizeof(T), 9, 9, 0, 9, true))
                push!(cfgs, KernelConfig(262144, 1, 512, 1, 1024 * sizeof(T), 9, 18, 0, 10, false))
            end

            if log2n < 25
                compiledKernels = map(params -> compile_kernel(params, log2n, reducer), cfgs)
            else
                compiledKernels = Function[]
                push!(compiledKernels, compile_kernel(cfgs[1], log2n, reducer))
                push!(compiledKernels, compile_kernel(cfgs[2], log2n, reducer))
                push!(compiledKernels, compile_kernel(cfgs[3], log2n, reducer, false))
            end
            
            return new{T}(n, p, reducer, npru, log2n, rootOfUnityTable, compiledKernels)
        else
            throw("Ring size not supported yet.")
        end

        return new{T}(Int32(n), p, reducer, npru, log2n, rootOfUnityTable)
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

function compile_kernel(params::KernelConfig, log2n::Int32, modulus::Reducer{T}, standard::Bool = true) where T<:Unsigned
    temp = CUDA.zeros(T, 1)
    shmem_length = Int32(params.shared_memory ÷ sizeof(T))

    # @device_code_ptx kernel = @cuda launch=false ntt_kernel1!(temp, temp, temp, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
    # println(occupancy(kernel.fun, 1024; shmem = 1024 * 2 * sizeof(T)))
    # throw("sigma")

    if standard
        kernel = @cuda launch=false ntt_kernel1!(temp, temp, temp, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
    else
        kernel = @cuda launch=false ntt_kernel2!(temp, temp, temp, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel)
    end

    func(in, out, rouTable) = kernel(in, out, rouTable, modulus, params.shared_index, params.logm, params.outer_iteration_count, log2n, shmem_length, params.not_last_kernel; threads = (params.blockdim_x, params.blockdim_y), blocks = (params.griddim_x, params.griddim_y), shmem = params.shared_memory)

    return func
end

function plan_ntt(n::Integer, p::T, npru::T) where T<:Integer
    @assert ispow2(n) "n: $n"
    @assert isprime(p) "p: $p"
    # @assert is_primitive_root(npru, p, n)

    return NTTPlan(n, p, npru)
end

function ntt!(vec::CuVector{T}, plan::NTTPlan{T}, bitreversedoutput = false) where T<:Unsigned
    @assert intlog2(length(vec)) == plan.log2len

    if plan.log2len < 12
        return old_ntt!(vec, plan, bitreversedoutput)
    end

    if plan.rootOfUnityTable isa Vector{T}
        curoutable = CuArray(plan.rootOfUnityTable)
        for kernel in plan.compiledKernels
            kernel(vec, vec, curoutable)
        end
    else
        for kernel in plan.compiledKernels
            kernel(vec, vec, plan.rootOfUnityTable)
        end
    end
    
    if !bitreversedoutput
        correct = parallel_bit_reverse_copy(vec)
        vec .= correct
    end

    return nothing
end