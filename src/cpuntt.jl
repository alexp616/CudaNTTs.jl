# Defining a new implementation
# To define a new FFT implementation in your own module, you should

# Define a new subtype (e.g. MyPlan) of AbstractFFTs.Plan{T} for FFTs and related transforms on arrays of T. This must have a pinv::Plan field, initially undefined when a MyPlan is created, that is used for caching the inverse plan.

# Define a new method AbstractFFTs.plan_fft(x, region; kws...) that returns a MyPlan for at least some types of x and some set of dimensions region. The region (or a copy thereof) should be accessible via fftdims(p::MyPlan) (which defaults to p.region), and the input size size(x) should be accessible via size(p::MyPlan).

# Define a method of LinearAlgebra.mul!(y, p::MyPlan, x) that computes the transform p of x and stores the result in y.

# Define a method of *(p::MyPlan, x), which can simply call your mul! method. This is not defined generically in this package due to subtleties that arise for in-place and real-input FFTs.

# If the inverse transform is implemented, you should also define plan_inv(p::MyPlan), which should construct the inverse plan to p, and plan_bfft(x, region; kws...) for an unnormalized inverse ("backwards") transform of x. Implementations only need to provide the unnormalized backwards FFT, similar to FFTW, and we do the scaling generically to get the inverse FFT.

# You can also define similar methods of plan_rfft and plan_brfft for real-input FFTs.

# To support adjoints in a new plan, define the trait AbstractFFTs.AdjointStyle. AbstractFFTs implements the following adjoint styles: AbstractFFTs.FFTAdjointStyle, AbstractFFTs.RFFTAdjointStyle, AbstractFFTs.IRFFTAdjointStyle, and AbstractFFTs.UnitaryAdjointStyle. To define a new adjoint style, define the methods AbstractFFTs.adjoint_mul and AbstractFFTs.output_size.

# multidimensional


# 

function slow_ntt(vec, npru::T, p::T) where T<:Integer
    n = length(vec)
    if !is_primitive_root(npru, p, n)
        throw(ArgumentError("$npru must be a primitive $n-th root of unity of p"))
    end
    mat = zeros(T, n, n)

    w = 1
    for r in 1:n
        temp = 1
        for c in 1:r
            mat[r, c] = temp
            temp = mul_mod(temp, w, p)
        end
        w = mul_mod(w, npru, p)
    end

    for r in 1:n
        for c in r+1:n
            mat[r, c] = mat[c, r]
        end
    end
    
    # temporary solution
    mat = BigInt.(mat)
    temp = BigInt.(vec)
    
    vec .= T.((mat * temp) .% p)

    return vec
end

function get_transposed_index(idx::T, rows::T, cols::T) where T<:Integer
    originalRow = idx % rows
    originalCol = idx ÷ rows

    result = originalCol + originalRow * cols

    return result
end

function final_transpose(idx::Integer, bitlength::Integer, numsPerBlock::Integer, lastFFTLen::Integer)
    firstswaplength = intlog2(lastFFTLen)
    unchangedbitslen = intlog2(numsPerBlock ÷ lastFFTLen)
    middlebitslen = bitlength - 2 * firstswaplength - unchangedbitslen

    lastBits = idx & ((1 << firstswaplength) - 1)
    idx >>= firstswaplength
    unchangedbits = idx & ((1 << unchangedbitslen) - 1)
    idx >>= unchangedbitslen
    middlebits = idx & ((1 << middlebitslen) - 1)
    idx >>= middlebitslen
    firstBits = idx & ((1 << firstswaplength) - 1)
    
    middlebits = digit_reverse(middlebits, numsPerBlock, middlebitslen ÷ intlog2(numsPerBlock))
    offset = firstswaplength

    result = firstBits
    result |= unchangedbits << offset
    offset += unchangedbitslen
    result |= middlebits << offset
    offset += middlebitslen
    result |= lastBits << offset

    return typeof(idx)(result)
end

# Defining a new implementation
# To define a new FFT implementation in your own module, you should

# Define a new subtype (e.g. MyPlan) of AbstractFFTs.Plan{T} for FFTs and related transforms on arrays of T. This must have a pinv::Plan field, initially undefined when a MyPlan is created, that is used for caching the inverse plan.

# Define a new method AbstractFFTs.plan_fft(x, region; kws...) that returns a MyPlan for at least some types of x and some set of dimensions region. The region (or a copy thereof) should be accessible via fftdims(p::MyPlan) (which defaults to p.region), and the input size size(x) should be accessible via size(p::MyPlan).

# Define a method of LinearAlgebra.mul!(y, p::MyPlan, x) that computes the transform p of x and stores the result in y.

# Define a method of *(p::MyPlan, x), which can simply call your mul! method. This is not defined generically in this package due to subtleties that arise for in-place and real-input FFTs.

# If the inverse transform is implemented, you should also define plan_inv(p::MyPlan), which should construct the inverse plan to p, and plan_bfft(x, region; kws...) for an unnormalized inverse ("backwards") transform of x. Implementations only need to provide the unnormalized backwards FFT, similar to FFTW, and we do the scaling generically to get the inverse FFT.

# You can also define similar methods of plan_rfft and plan_brfft for real-input FFTs.

# To support adjoints in a new plan, define the trait AbstractFFTs.AdjointStyle. AbstractFFTs implements the following adjoint styles: AbstractFFTs.FFTAdjointStyle, AbstractFFTs.RFFTAdjointStyle, AbstractFFTs.IRFFTAdjointStyle, and AbstractFFTs.UnitaryAdjointStyle. To define a new adjoint style, define the methods AbstractFFTs.adjoint_mul and AbstractFFTs.output_size.

# multidimensional

struct CPUNTTPlan{T}
    n::Int
    p::T
    npru::T
    numsPerThread::Int
    threadsPerBlock::Int
    numsPerBlock::Int
    numBlocks::Int
    numIterations::Int
    lastFFTLen::Int
    npbpruPowerTable::Array{T}
end

function plan_cpuntt(n::Int, p::T, npru::T; numsPerThread = 2, threadsPerBlock = 1) where T<:Integer
    @assert ispow2(n)

    numsPerBlock = numsPerThread * threadsPerBlock
    numBlocks = n ÷ numsPerBlock
    numIterations = Int(ceil(log(numsPerBlock, n)))
    lastFFTLen = n ÷ numsPerBlock ^ (numIterations - 1)
    npbpruPowerTable = generate_twiddle_factors(powermod(npru, n ÷ numsPerBlock, p), p, numsPerBlock)

    return CPUNTTPlan{T}(
        n,
        p,
        npru,
        numsPerThread,
        threadsPerBlock,
        numsPerBlock,
        numBlocks,
        numIterations,
        lastFFTLen,
        npbpruPowerTable
    )
end

function ntt(vec::Vector{T}, p::T, npru::T) where T<:Integer
    @assert ispow2(length(vec))
    plan = plan_ntt(length, p, npru)

    return ntt(vec, plan)
end

function ntt(vec::Vector{T}, plan::CPUNTTPlan{T}) where T<:Integer
    @assert length(vec) == plan.n

    n = plan.n
    p = plan.p
    npru = plan.npru
    numsPerThread = plan.numsPerThread
    threadsPerBlock = plan.threadsPerBlock
    numsPerBlock = plan.numsPerBlock
    numBlocks = plan.numBlocks
    numIterations = plan.numIterations
    lastFFTLen = plan.lastFFTLen
    npbpruPowerTable = plan.npbpruPowerTable
    log2n = intlog2(n)

    original = vec
    aux = zeros(T, n)

    shared = zeros(T, numsPerBlock)

    twiddleStride = 1
    for _ in 1:numIterations - 1
        fftLen = numsPerBlock
        log2fftLen = intlog2(fftLen)

        threadStride = numBlocks
        for blockIdx in 1:numBlocks
            for threadIdx in 1:threadsPerBlock
                fftIdx = threadIdx - 1
                globalIdx = blockIdx + fftIdx * threadStride
                globalIdxStride = threadStride * threadsPerBlock
                for _ in 1:numsPerThread
                    val = vec[globalIdx]
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    shared[bitreversed + 1] = val
                    fftIdx += threadsPerBlock
                    globalIdx += globalIdxStride
                end
            end

            for i in 1:log2fftLen
                m = 1 << i
                m2 = m >> 1
                bits = log2fftLen - i
                mask = (1 << bits) - 1

                for threadIdx in 1:threadsPerBlock
                    threadStart = (threadIdx - 1) * (numsPerThread >> 1)

                    for j in 0:(numsPerThread >> 1) - 1
                        gh_j = j + threadStart
                        idx1 = m * (gh_j & mask) + (gh_j >> bits)
                        idx2 = idx1 + m2
                        theta = npbpruPowerTable[((fftLen >> i) * (gh_j >> bits)) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)
                    end
                end
            end
            
            blockTwiddleStride = ((blockIdx - 1) ÷ twiddleStride) * twiddleStride
            for threadIdx in 1:threadsPerBlock
                fftIdx = threadIdx - 1
                globalIdx = fftIdx + (blockIdx - 1) * numsPerBlock
                for _ in 1:numsPerThread
                    shared[fftIdx + 1] = mul_mod(shared[fftIdx + 1], powermod(npru, fftIdx * blockTwiddleStride, p), p)
                    aux[globalIdx + 1] = shared[fftIdx + 1]
                    globalIdx += threadsPerBlock
                    fftIdx += threadsPerBlock
                end
            end
        end

        temp = aux
        aux = vec
        vec = temp

        twiddleStride *= numsPerBlock
    end

    for _ in 1:1
        fftLen = lastFFTLen
        log2fftLen = intlog2(fftLen)
        fftsPerBlock = numsPerBlock ÷ lastFFTLen

        threadStride = numBlocks
        for blockIdx in 1:numBlocks
            for threadIdx in 1:threadsPerBlock
                virtualThreadIdx = threadIdx - 1
                globalIdx = blockIdx + virtualThreadIdx * threadStride
                added = threadStride * threadsPerBlock
                for _ in 1:numsPerThread
                    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, lastFFTLen)
                    fftNum = transposedIdx ÷ fftLen
                    val = vec[globalIdx]

                    fftIdx = transposedIdx % fftLen
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    
                    shared[fftNum * fftLen + bitreversed + 1] = val
                    virtualThreadIdx += threadsPerBlock
                    globalIdx += added
                end
            end

            for i in 1:log2fftLen
                m = 1 << i
                m2 = m >> 1
                bits = log2fftLen - i
                mask = (1 << bits) - 1

                for threadIdx in 1:threadsPerBlock
                    offset = 0
                    for j in 0:(numsPerThread >> 1) - 1
                        virtualThreadIdx = threadIdx + offset - 1
                        gh_j = virtualThreadIdx % (fftLen >> 1)
                        fftNum = virtualThreadIdx ÷ (fftLen >> 1)
                        idx1 = m * (gh_j & mask) + (gh_j >> bits) + (fftLen * fftNum)
                        idx2 = idx1 + m2
                        theta = npbpruPowerTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)

                        offset += threadsPerBlock
                    end
                end
            end

            for threadIdx in 1:threadsPerBlock
                fftIdx = threadIdx - 1
                globalIdx = fftIdx + (blockIdx - 1) * numsPerBlock
                for _ in 1:numsPerThread
                    aux[final_transpose(Int32(globalIdx), log2n, Int32(numsPerBlock), Int32(lastFFTLen)) + 1] = shared[fftIdx + 1]
                    globalIdx += threadsPerBlock
                    fftIdx += threadsPerBlock
                end
            end
        end
    end

    if pointer(aux) != pointer(original)
        original .= aux
    end

    return nothing
end