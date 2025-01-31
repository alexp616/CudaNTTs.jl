struct NTTPlan{T<:Unsigned}
    n::Int32
    p::T
    reducer::Reducer{T}
    npru::T
    log2len::Int32
    rootOfUnityTable::CuVector{T}

    function NTTPlan(n::Integer, p::T, npru::T) where T<:Integer
        reducer = BarrettReducer(p)
        rootOfUnityTable = gpu_root_of_unity_table_generator(modsqrt(npru, p), p, n)

        return new{T}(Int32(n), p, reducer, npru, Int32(intlog2(n)), rootOfUnityTable)
    end
end

function plan_ntt(n::Integer, p::T, npru::T) where T<:Integer
    @assert ispow2(n) "n: $n"
    @assert isprime(p) "p: $p"
    @assert is_primitive_root(npru, p, n)

    return NTTPlan(n, p, npru)
end


# three sections:
# 0 -> 11
# 12 -> 29

# in-place
# out-of-place, bit reversed
# out-of-place, correct

function ntt!(vec::CuVector{T}, plan::NTTPlan{T}) where T<:Unsigned
    if plan.log2len != 12 || length(vec) != 2^12
        throw("skibidi")
    end
    @cuda threads = (64, 4) blocks = (8, 1) shmem = 512 * sizeof(T) ntt_kernel1!(vec, vec, plan.rootOfUnityTable, plan.reducer, Int32(8), Int32(0), Int32(3), plan.log2len, true)

    @cuda threads = (256, 1) blocks = (1, 8) shmem = 512 * sizeof(T) ntt_kernel1!(vec, vec, plan.rootOfUnityTable, plan.reducer, Int32(8), Int32(3), Int32(9), plan.log2len, false)

    return nothing
end