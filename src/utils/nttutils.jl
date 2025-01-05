include("modoperations.jl")
include("modsqrt.jl")
include("modcubert.jl")

function intlog2(x::Int64)
    return 64 - leading_zeros(x - 1)
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

function is_primitive_root(n::T, p::T, order::Int) where T<:Integer
    temp = n
    for i in 1:order - 1
        if temp == 1
            return false
        end

        temp = mul_mod(temp, n, p)
    end

    return temp == 1
end

function primitive_nth_root_of_unity(n::Int, p::Integer)
    if (p - 1) % n != 0
        throw("n must divide p - 1")
    end
    temp, a, b = find_next_2a3b(n)
    if temp != n
        throw("This method only works for $n that can be written as 2^a*3^b")
    end
    
    if a > 0
        g = p - 1 # p - 1 always a 2nd root of unity
        a -= 1
    else
        g = 1
    end

    while a > 0
        a -= 1
        g = modsqrt(g, p)
    end

    while b > 0
        b -= 1
        g = modcubert(g, p)
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

function bit_reverse(x::Int, log2n::Int)
    temp = 0
    for i in 1:log2n
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

function digit_reverse(x::Int, base::Int, logn)
    temp = 0

    for i in 1:logn
        x, b = divrem(x, base)
        temp = temp * base + b
    end
    
    return temp
end