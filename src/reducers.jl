abstract type Reducer{T<:Integer} end

"""
    BarrettReducer{T<:Union{UInt32, UInt64, Int32, Int64}}

Struct holding constants needed for barrett reduction.
"""
struct BarrettReducer{T<:Integer} <: Reducer{T}
    p::T
    km2::Int
    kp3::Int
    μ::T

    function BarrettReducer(p::T) where T<:Integer
        @assert p < typemax(T) >>> 2
        k = Int(ceil(log2(p)))
        μ = T(fld(BigInt(1) << (2*k + 1), p))
        return new{T}(p, k-2, k+3, μ)
    end
end

@inline function add_mod(x::T, y::T, m::BarrettReducer{T})::T where T<:INTTYPES
    result = x + y
    return (result >= m.p || result < x) ? result - m.p : result
end

@inline function sub_mod(x::T, y::T, m::BarrettReducer{T})::T where T<:INTTYPES
    if y > x
        return (m.p - y) + x
    else
        return x - y
    end
end

function mul_mod(a::T, b::T, reducer::BarrettReducer{T})::T where T<:INTTYPES
    C = mywidemul(a, b)
    p = reducer.p

    r = (C >>> reducer.km2)
    r *= reducer.μ
    r >>>= reducer.kp3
    r *= p
    Cout = C - r
    Cout = Cout >= p ? Cout - p : Cout
    return unsafe_trunc(T, Cout)
end

function power_mod(n::T, p::Integer, m::BarrettReducer{T}) where T<:Integer
    result = one(T)
    base = n

    z = zero(typeof(p))
    on = one(typeof(p))
    while p > z
        if p & on == on
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        p = p >>> 1
    end

    return result
end

function br_power_mod(n::T, pow::Int32, log2n::Int32, m::BarrettReducer{T}) where T<:Integer
    result = one(T)
    base = n

    mask = o << (log2n - o)
    
    for i in o:log2n
        if pow & mask != zero(Int32)
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        mask >>>= 1
    end

    return result
end