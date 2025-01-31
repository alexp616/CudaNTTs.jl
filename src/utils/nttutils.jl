include("modoperations.jl")
include("modsqrt.jl")

function intlog2(x::Int64)
    return 64 - leading_zeros(x - 1)
end

function intlog2(x::Int32)::Int32
    return Int32(32) - leading_zeros(x - Int32(1))
end

function is_primitive_root(npru::T, p::T, order::Integer) where T<:Integer
    npru = BigInt(npru)
    p = BigInt(p)
    temp = npru
    for i in 1:order - 1
        if temp == 1
            return false
        end

        temp = mul_mod(temp, npru, p)
    end

    return temp == 1
end

"""
    primitive_nth_root_of_unity(n::Integer, p::Integer)

Return a primitive n-th root of unity of the field 𝔽ₚ
"""
function primitive_nth_root_of_unity(n::Integer, p::Integer)
    @assert ispow2(n)
    if (p - 1) % n != 0
        throw("n must divide p - 1")
    end

    g = p - typeof(p)(1)

    a = intlog2(n)

    while a > 1
        a -= 1
        original = g
        g = modsqrt(g, p)
        @assert powermod(g, 2, p) == original
    end

    # @assert is_primitive_root(g, p, n)
    return g
end

function gpu_root_of_unity_table_generator(npru::T, p::Reducer{T}, n::Integer) where T<:Integer
    return CuArray(root_of_unity_table_generator(npru, p, n))
end

"""
    root_of_unity_table_generator(npru::T, p::T, n::Integer) where T<:Integer

Returns array containing powers 0 -> n-1 of npru mod p. Accessed as:
arr[i] = npru ^ (i - 1)
"""
function root_of_unity_table_generator(npru::T, p::Reducer{T}, n::Integer) where T<:Integer
    # @assert is_primitive_root(npru, p, n)

    result = zeros(T, n)
    curr = T(1)
    for i in eachindex(result)
        result[i] = curr
        curr = mul_mod(curr, npru, p)
    end

    bit_reverse_vector(result)

    return result
end

function find_ntt_primes(len::Int, T = UInt32, num = 10)
    len *= 2
    prime_list = []
    k = fld(typemax(T) >> 2, len)
    while length(prime_list) < num
        candidate = k * len + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k -= 1

        if k < 0
            throw("Not enough primes found. Primes found: $(prime_list)")
        end
    end

    return prime_list
end

function find_ntt_prime(len::Int, T::DataType)
    len *= 2
    k = fld(typemax(T) >> 2, len)
    while true
        candidate = T(k * len + 1)
        if isprime(candidate)
            return candidate
        end
        k -= 1
        if k <= 0
            throw("No NTT prime found")
        end
    end
end

function bit_reverse(x::T, log2n::T)::T where T<:Integer
    temp = zero(T)
    for _ in one(T):log2n
        temp <<= one(T)
        temp |= (x & one(T))
        x >>= one(T)
    end
    return temp
end

function parallel_bit_reverse_copy(vec::CuVector)
    @assert ispow2(length(vec))
    
    log2n = intlog2(length(vec))
    dest = CUDA.zeros(eltype(vec), length(vec))

    kernel = @cuda launch=false parallel_bit_reverse_copy_kernel(vec, dest, log2n)
    config = launch_configuration(kernel.fun)
    threads = min(length(vec), Base._prevpow2(config.threads))
    blocks = div(length(vec), threads)

    CUDA.@sync kernel(vec, dest, log2n; threads = threads, blocks = blocks)

    return dest
end

function parallel_bit_reverse_copy_kernel(src::CuDeviceVector, dest::CuDeviceVector, log2n::Int)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds dest[bit_reverse(idx - 1, log2n) + 1] = src[idx]

    return nothing
end