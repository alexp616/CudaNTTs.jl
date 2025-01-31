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

function cpu_ntt!(vec::Vector{T}, plan::NTTPlan{T}, bitreversedoutput = false) where T<:Integer
    @assert length(vec) == plan.n

    if !bitreversedoutput
        bit_reverse_vector(vec)
    end

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

    return nothing
end

# function cpu_intt!(vec::Vector{T}, plan::INTTPlan{T}) where T<:Integer
#     @assert length(vec) == plan.n
#     # bit_reverse_vector(vec)

#     log2n = intlog2(plan.n)

#     @inbounds begin
#     for s in 1:log2n
#         m = 1 << s
#         m2 = m >> 1
#         ωₘ = powermod(plan.npru, plan.n ÷ m, plan.p)
#         for k in 0:m:plan.n-1
#             ω = one(T)
#             for j in 1:m2
#                 t = mul_mod(ω, vec[k + j + m2], plan.p)
#                 u = vec[k + j]
#                 vec[k + j] = add_mod(u, t, plan.p)
#                 vec[k + j + m2] = sub_mod(u, t, plan.p)
#                 ω = mul_mod(ω, ωₘ, plan.p)
#             end
#         end
#     end

#     for i in eachindex(vec)
#         vec[i] = mul_mod(vec[i], plan.lenInverse, plan.p)
#     end
#     end

#     return nothing
# end