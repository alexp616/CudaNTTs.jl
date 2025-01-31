function generate_theta_m(p::T, len, log2len, npru::T) where T<:Integer
    result = zeros(T, log2len)
    for i in 1:log2len
        m = 1 << i
        result[i] = powermod(npru, len ÷ m, p)
    end

    return result
end

function old_ntt!(vec::CuVector{T}, plan::NTTPlan{T}, bitreversedoutput = false) where T<:Unsigned
    if !bitreversedoutput
        correct = parallel_bit_reverse_copy(vec)
        vec .= correct
    end

    kernel = @cuda launch=false old_ntt_kernel!(vec, plan.reducer, T(0), 0, 0, 0, 0)
    config = launch_configuration(kernel.fun)
    threads = min(length(vec) >> 1, Base._prevpow2(config.threads))
    blocks = cld(length(vec) >> 1, threads)

    log2n = intlog2(length(vec))
    theta_array = generate_theta_m(plan.p, length(vec), plan.log2len, plan.npru)
    for i in 1:log2n
        m = 1 << i
        m2 = m >> 1
        magicbits = log2n - i
        magicmask = (1 << magicbits) - 1

        theta_m = theta_array[i]

        CUDA.@sync kernel(vec, plan.reducer, theta_m, magicmask, magicbits, m, m2; threads = threads, blocks = blocks)
        # println("vec after iteration $i: $vec \n")
    end

    

    return nothing
end

function old_ntt_kernel!(vec::CuDeviceArray{T}, modulus::Reducer{T}, theta_m::T, magicmask::Int, magicbits::Int, m::Int, m2::Int) where T<:Unsigned
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = m * (idx & magicmask) + (idx >> magicbits)

    @inbounds begin

    theta = power_mod(theta_m, idx >> magicbits, modulus)
    
    t = mul_mod(theta, vec[k + m2 + 1], modulus)
    u = vec[k + 1]

    vec[k + 1] = add_mod(u, t, modulus)
    vec[k + m2 + 1] = sub_mod(u, t, modulus)

    end

    return nothing
end