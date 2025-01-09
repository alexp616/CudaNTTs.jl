include("modoperations.jl")
include("modsqrt.jl")
include("modcubert.jl")

function intlog2(x::Int64)
    return 64 - leading_zeros(x - 1)
end

function intlog2(x::Int32)
    return 32 - leading_zeros(x - Int32(1))
end

function find_next_2a3b(n::Int)
    lowest = Base._nextpow2(n)
    lowpow2 = intlog2(lowest)
    lowpow3 = 0

    num = lowest
    pow2 = lowpow2
    pow3 = lowpow3

    while pow2 > 0
        num ÷= 2
        pow2 -= 1
        num *= 3
        pow3 += 1

        if (pow2 != 0) && (num ÷ 2 >= n)
            num ÷= 2
            pow2 -= 1
        end

        if num < lowest
            lowest = num
            lowpow2 = pow2
            lowpow3 = pow3
        end
    end

    return lowest, lowpow2, lowpow3
end

function is_primitive_root(npru::T, p::T, order::Integer) where T<:Integer
    temp = npru
    for i in 1:order - 1
        if temp == 1
            return false
        end

        temp = mul_mod(temp, npru, p)
    end

    return temp == 1
end

function primitive_nth_root_of_unity(n::Int, p::Integer)
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

    @assert is_primitive_root(g, p, n)
    return g
end

"""
    generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer

Returns array containing powers 0 -> n-1 of npru mod p. Accessed as:
arr[i] = npru ^ (i - 1)
"""
function generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer
    @assert is_primitive_root(npru, p, n)

    result = zeros(T, n)
    curr = T(1)
    for i in eachindex(result)
        result[i] = curr
        curr = mul_mod(curr, npru, p)
    end

    return result
end

function find_ntt_primes(len::Int, T = UInt32, num = 10)
    prime_list = []
    k = fld(typemax(T), len)
    while length(prime_list) < num
        candidate = k * len + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k -= 1
    end

    return prime_list
end

function bit_reverse(x::Integer, log2n::Integer)
    temp = zero(typeof(x))
    for i in 1:log2n
        temp <<= typeof(x)(1)
        temp |= (x & typeof(x)(1))
        x >>= typeof(x)(1)
    end
    return temp
end

function digit_reverse(x::Integer, base::Integer, logn::Integer)
    temp = 0

    for _ in 1:logn
        x, b = divrem(x, base)
        temp = temp * base + b
    end
    
    return temp
end