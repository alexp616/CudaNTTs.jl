struct NTTPlan{T}
    n::Int
    p::T
    npru::T
    threadsPerBlock::Int
    numsPerBlock::Int
    numBlocks::Int
    numIterations::Int
    lastFFTLen::Int
    npbpruPowerTable::CuVector{T}
end

function plan_ntt(n::Integer, p::T, npru::T) where T<:Integer
    @assert ispow2(n)
    @assert isprime(p)
    @assert is_primitive_root(npru, p, n)

    threadsPerBlock = 64
    numsPerBlock = 2 * threadsPerBlock
    numBlocks = n ÷ numsPerBlock
    numIterations = Int(ceil(log(numsPerBlock, n)))
    lastFFTLen = n ÷ numsPerBlock ^ (numIterations - 1)
    npbpruPowerTable = CuArray(generate_twiddle_factors(powermod(npru, n ÷ numsPerBlock, p), p, numsPerBlock))

    return NTTPlan{T}(
        n,
        p,
        npru,
        threadsPerBlock,
        numsPerBlock,
        numBlocks,
        numIterations,
        lastFFTLen,
        npbpruPowerTable,
        # kernel1,
        # kernel2
    )
end

function ntt!(vec::CuVector{T}, plan::NTTPlan{T}) where T<:Integer
    @assert length(vec) == plan.n

    n = plan.n
    p = plan.p
    npru = plan.npru
    threadsPerBlock = plan.threadsPerBlock
    numsPerBlock = plan.numsPerBlock
    numBlocks = plan.numBlocks
    numIterations = plan.numIterations
    lastFFTLen = plan.lastFFTLen
    npbpruPowerTable = plan.npbpruPowerTable
    twiddleStride = 1

    original = vec
    aux = CUDA.zeros(T, n)

    kernel1 = @cuda launch=false ntt_kernel1!(vec, p, npru, aux, npbpruPowerTable, numsPerBlock, twiddleStride)
    kernel2 = @cuda launch=false ntt_kernel2!(vec, p, aux, npbpruPowerTable, numsPerBlock, lastFFTLen)

    shmemSize = length(npbpruPowerTable) * sizeof(T)
    
    for _ in 1:numIterations - 1
        CUDA.@sync kernel1(vec, p, npru, aux, npbpruPowerTable, numsPerBlock, twiddleStride; threads = threadsPerBlock, blocks = numBlocks, shmem = shmemSize)
        twiddleStride *= numsPerBlock

        temp = aux
        aux = vec
        vec = temp
    end

    CUDA.@sync kernel2(vec, p, aux, npbpruPowerTable, numsPerBlock, lastFFTLen; threads = threadsPerBlock, blocks = numBlocks, shmem = shmemSize)

    if pointer(aux) != pointer(original)
        original .= aux
    end

    CUDA.unsafe_free!(aux)

    return nothing
end

function ntt_kernel1!(vec::CuDeviceVector{T}, p::T, npru::T, aux::CuDeviceVector{T}, npbpruPowerTable::CuDeviceVector{T}, numsPerBlock::Int, twiddleStride::Int) where T<:Integer
    log2FFTLen = intlog2(numsPerBlock)

    shared = @cuDynamicSharedMem(T, numsPerBlock)
    
    # Load nums into shared memory
    fftIdx = threadIdx().x - 1
    globalIdx = blockIdx().x + fftIdx * gridDim().x

    val = vec[globalIdx]
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[bitreversed + 1] = val

    fftIdx += blockDim().x
    globalIdx += gridDim().x * blockDim().x

    val = vec[globalIdx]
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[bitreversed + 1] = val

    # FFT in shared memory

    CUDA.sync_threads()
    k = threadIdx().x - 1
    for i in 1:log2FFTLen
        m = 1 << i        
        m2 = m >> 1
        bits = log2FFTLen - i
        mask = (1 << bits) - 1
        
        idx1 = m * (k & mask) + (k >> bits) + 1
        idx2 = idx1 + m2
        theta = npbpruPowerTable[((numsPerBlock >> i) * (k >> bits)) + 1]
        t = mul_mod(theta, shared[idx2], p)
        u = shared[idx1]

        shared[idx1] = add_mod(u, t, p)
        shared[idx2] = sub_mod(u, t, p)

        CUDA.sync_threads()
    end

    # Apply twiddle and write to output array
    blockTwiddleStride = ((blockIdx().x - 1) ÷ twiddleStride) * twiddleStride
    
    fftIdx = threadIdx().x
    globalIdx = fftIdx + (blockIdx().x - 1) * numsPerBlock

    shared[fftIdx] = mul_mod(shared[fftIdx], power_mod(npru, (fftIdx - 1) * blockTwiddleStride, p), p)
    aux[globalIdx] = shared[fftIdx]

    globalIdx += blockDim().x
    fftIdx += blockDim().x

    shared[fftIdx] = mul_mod(shared[fftIdx], power_mod(npru, (fftIdx - 1) * blockTwiddleStride, p), p)
    aux[globalIdx] = shared[fftIdx]

    return nothing
end

function ntt_kernel2!(vec::CuDeviceVector{T}, p::T, aux::CuDeviceVector{T}, npbpruPowerTable::CuDeviceVector{T}, numsPerBlock, fftLen::Int) where T<:Integer
    shared = @cuDynamicSharedMem(T, numsPerBlock)
    log2FFTLen = intlog2(fftLen)

    fftsPerBlock = numsPerBlock ÷ fftLen

    virtualThreadIdx = threadIdx().x - 1
    globalIdx = blockIdx().x + virtualThreadIdx * gridDim().x
    globalIdxStride = gridDim().x * blockDim().x
    
    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx ÷ fftLen
    fftIdx = transposedIdx % fftLen

    bitreversed = bit_reverse(fftIdx, log2FFTLen)

    shared[fftNum * fftLen + bitreversed + 1] = val
    virtualThreadIdx += blockDim().x
    globalIdx += globalIdxStride

    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx ÷ fftLen
    fftIdx = transposedIdx % fftLen

    bitreversed = bit_reverse(fftIdx, log2FFTLen)

    shared[fftNum * fftLen + bitreversed + 1] = val
    CUDA.sync_threads()

    for i in 1:log2FFTLen
        m = 1 << i
        m2 = m >> 1
        bits = log2FFTLen - i
        mask = (1 << bits) - 1

        k = threadIdx().x - 1
        gh_j = k % (fftLen >> 1)
        fftNum = k ÷ (fftLen >> 1)
        idx1 = m * (gh_j & mask) + (gh_j >> bits) + (fftLen * fftNum) + 1
        idx2 = idx1 + m2
        theta = npbpruPowerTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + 1]
        t = mul_mod(theta, shared[idx2], p)
        u = shared[idx1]

        shared[idx1] = add_mod(u, t, p)
        shared[idx2] = sub_mod(u, t, p)
        
        CUDA.sync_threads()
    end

    fftIdx = threadIdx().x - 1
    globalIdx = fftIdx + (blockIdx().x - 1) * numsPerBlock
    log2n = intlog2(length(vec))

    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + 1] = shared[fftIdx + 1]
    globalIdx += blockDim().x
    fftIdx += blockDim().x

    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + 1] = shared[fftIdx + 1]

    return nothing
end