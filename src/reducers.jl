abstract type Reducer{T<:INTTYPES} end

"""
    BarrettReducer{T<:Union{UInt32, UInt64, Int32, Int64}}

Struct holding constants needed for barrett reduction.
"""
struct BarrettReducer{T<:INTTYPES} <: Reducer{T}
    p::T
    k::Int
    μ::T

    function BarrettReducer(p::T) where T<:INTTYPES
        @assert p < typemax(T) >> 2
        k = Int(ceil(log2(p)))
        μ = T(fld(BigInt(1) << (2*k + 1), p))
        return new{T}(p, k, μ)
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

@inline function mul_mod(a::T, b::T, reducer::BarrettReducer{T})::T where T<:INTTYPES
    C = mywidemul(a, b)
    p = reducer.p
    μ = reducer.μ
    k = reducer.k

    r = (C >> (k - 2))
    r *= μ
    r >>= (k + 3)
    r *= p
    Cout = C - r
    Cout = Cout >= p ? Cout - p : Cout
    return unsafe_trunc(T, Cout)
end

function power_mod(n::T, p::Integer, m::BarrettReducer{T}) where T<:INTTYPES
    result = one(T)
    base = n

    z = zero(typeof(p))
    on = one(typeof(p))
    while p > z
        if p & on == on
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        p = p >> 1
    end

    return result
end