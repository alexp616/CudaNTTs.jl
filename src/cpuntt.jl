function bit_reverse_vector(src::Vector{T}) where T<:Integer
    @assert ispow2(length(src))
    aux = zeros(T, length(src))

    log2n = intlog2(length(src))
    for i in eachindex(src)
        val = src[i]
        bitreversed = bit_reverse(i - 1, log2n)
        aux[bitreversed + 1] = val
    end

    src .= aux
end

function cpu_ntt!(vec::Vector{T}, plan::NTTPlan{T}, bitreversedoutput::Bool = false) where T<:Integer
    if length(vec) != plan.n
        throw(ArgumentError("Plan is for length $(plan.n), input vector is length $(length(vec))"))
    end

    bit_reverse_vector(vec)

    for s in 1:plan.log2len
        m = 1 << s
        m2 = m >> 1
        ωₘ = powermod(plan.npru, plan.n ÷ m, plan.p)
        for k in 0:m:plan.n-1
            ω = one(T)
            for j in 1:m2
                t = mul_mod(ω, vec[k + j + m2], plan.p)
                u = vec[k + j]
                vec[k + j] = add_mod(u, t, plan.p)
                vec[k + j + m2] = sub_mod(u, t, plan.p)
                ω = mul_mod(ω, ωₘ, plan.p)
            end
        end
        # println("vec after iteration $s: $vec \n")
    end

    if bitreversedoutput
        bit_reverse_vector(vec)
    end

    return nothing
end

function cpu_intt!(vec::Vector{T}, plan::INTTPlan{T}, bitreversedinput::Bool = false) where T<:Integer
    if length(vec) != plan.n
        throw(ArgumentError("Plan is for length $(plan.n), input vector is length $(length(vec))"))
    end

    if !bitreversedinput
        bit_reverse_vector(vec)
    end

    log2n = intlog2(plan.n)

    @inbounds begin
    for s in 1:log2n
        m = 1 << s
        m2 = m >> 1
        ωₘ = powermod(plan.npruinv, plan.n ÷ m, plan.p)
        for k in 0:m:plan.n-1
            ω = one(T)
            for j in 1:m2
                t = mul_mod(ω, vec[k + j + m2], plan.p)
                u = vec[k + j]
                vec[k + j] = add_mod(u, t, plan.p)
                vec[k + j + m2] = sub_mod(u, t, plan.p)
                ω = mul_mod(ω, ωₘ, plan.p)
            end
        end
    end

    for i in eachindex(vec)
        vec[i] = mul_mod(vec[i], plan.n_inverse, plan.p)
    end
    end

    return nothing
end