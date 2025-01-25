abstract type Reducer{T<:Unsigned} end

struct BarrettReducer{T<:Unsigned} <: Reducer
    p::T
    k::Int
    μ::T

    function BarrettReducer(p::T) where T<:Unsigned
        @assert p < typemax(T) >> 2
        k = Int(ceil(log2(p)))
        μ = T(fld(BigInt(1) << (2*k + 1), p))
        return new{T}(p, k, μ)
    end
end

function add_mod(x::T, y::T, reducer::BarrettReducer{T}) where T<:Unsigned
    result = x + y
    m = reducer.p
    return (result >= m || result < x) ? result - m : result
end

function sub_mod(x::T, y::T, m::BarrettReducer{T}) where T<:Unsigned
    if y > x
        return (m.p - y) + x
    else
        return x - y
    end
end

function mul_mod(a::T, b::T, reducer::BarrettReducer{T}) where T<:Unsigned
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
    return T(Cout)
end

function power_mod(n::T, p::Integer, m::BarrettReducer{T}) where T<:Unsigned
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