function sub_mod(x::T, y::T, m::T) where T<:Integer
    if y > x
        return (m - y) + x
    else
        return x - y
    end
end

function add_mod(x::T, y::T, m::T)::T where T<:Integer
    result = x + y
    return (result >= m || result < x) ? result - m : result
end

function mywiden(x)
    throw(MethodError(mywiden, (typeof(x),)))
end

macro generate_widen()
    int_types = [Int32, Int64, Int128]
    uint_types = [UInt32, UInt64, UInt128]

    widen_methods = quote end
    for i in 1:length(uint_types) - 1
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(int_types[i])) = $(int_types[i+1])(x)
        ))
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(uint_types[i])) = $(uint_types[i+1])(x)
        ))
    end

    return widen_methods
end

@generate_widen()

"""
    mywidemul(x::T, y::T) where T<:Integer

Exists because Base.widen() widens Int128 to BigInt, which 
CUDA doesn't like.
"""
@inline function mywidemul(x::T, y::T) where T<:Integer
    return mywiden(x) * mywiden(y)
end

function mul_mod(x::T, y::T, m::T) where T<:Integer
    return unsafe_trunc(T, mod(mywidemul(x, y), m))
end

function power_mod(n::T, p::Integer, m::T) where T<:Integer
    result = eltype(n)(1)
    base = mod(n, m)

    while p > 0
        if p & 1 == 1
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        p = p >> 1
    end

    return result
end

function br_power_mod(n::T, pow::Int32, log2n::Int32, m::T) where T<:Integer
    result = one(T)
    base = n

    mask = o << (log2n - o)

    for i in o:log2n
        if pow & mask != zero(Int32)
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        mask >>= 1
    end

    return result
end