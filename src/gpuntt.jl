global const o = Int32(1)

"""
    NTTPlan{T}

Stores all things needed to perform a NTT on a vector. Generated using
`plan_ntt()`
"""
struct NTTPlan{T}
    n::Int32
    p::T
    reducer::BarrettReducer{T}
    npru::T
    threadsPerBlock::Int32
    numsPerBlock::Int32
    numBlocks::Int32
    numIterations::Int32
    lastFFTLen::Int32
    npbpruPowerTable::CuVector{T}
    shmemSize::Int
    kernel1::CUDA.HostKernel
    kernel2::CUDA.HostKernel
end

"""
    NTTPlan{T}

Stores all things needed to perform a normalized INTT on a vector. Generated
using `plan_ntt()`
"""
struct INTTPlan{T}
    n::Int32
    p::T
    reducer::BarrettReducer{T}
    npru::T
    lenInverse::T
    threadsPerBlock::Int32
    numsPerBlock::Int32
    numBlocks::Int32
    numIterations::Int32
    lastFFTLen::Int32
    npbpruPowerTable::CuVector{T}
    shmemSize::Int
    kernel1::CUDA.HostKernel
    kernel2::CUDA.HostKernel
end

"""
    plan_ntt(n::Integer, p::T, npru::T) where T<:Integer

Generates both an NTT plan and an INTT plan. Returns both as a tuple.

# Arguments

- `n`: Length of the NTT. As of now and for the forseeable future, this 
has to be a power of 2.
- `p`: Characteristic of the field 𝔽ₚ to perform the NTT in.
- `npru`: A primitive n-th root of unity of 𝔽ₚ. The function `primitive_nth_root_of_unity()` 
provides a way to compute one quickly.
"""
function plan_ntt(n::Integer, p::T, npru::T) where T<:Unsigned
    @assert ispow2(n)
    @assert isprime(p)
    @assert is_primitive_root(npru, p, n)

    temp = CUDA.zeros(T, 1)
    reducer = BarrettReducer(p)
    kernel1 = @cuda launch=false ntt_kernel1!(temp, reducer, npru, temp, temp, Int32(0), Int32(0))
    kernel2 = @cuda launch=false ntt_kernel2!(temp, reducer, temp, temp, Int32(0), Int32(0))
    invkernel2 = @cuda launch=false ntt_kernel3!(temp, reducer, temp, temp, p, Int32(0), Int32(0))

    threadsPerBlock = min(n ÷ 2, Base._prevpow2(launch_configuration(invkernel2.fun).threads))
    numsPerBlock = 2 * threadsPerBlock
    numBlocks = n ÷ numsPerBlock
    numIterations = Int(ceil(log(numsPerBlock, n)))
    lastFFTLen = n ÷ numsPerBlock ^ (numIterations - 1)
    npbpruPowerTable = CuArray(generate_twiddle_factors(powermod(npru, n ÷ numsPerBlock, p), p, numsPerBlock))
    shmemSize = length(npbpruPowerTable) * sizeof(T) * 2

    nttPlan = NTTPlan{T}(
        Int32(n),
        p,
        reducer,
        npru,
        Int32(threadsPerBlock),
        Int32(numsPerBlock),
        Int32(numBlocks),
        Int32(numIterations),
        Int32(lastFFTLen),
        npbpruPowerTable,
        shmemSize,
        kernel1,
        kernel2
    )

    npruinv = invmod(npru, p)
    invnpbpruPowerTable = CuArray(generate_twiddle_factors(powermod(npruinv, n ÷ numsPerBlock, p), p, numsPerBlock))

    inttPlan = INTTPlan{T}(
        Int32(n),
        p,
        reducer,
        npruinv,
        T(invmod(n, p)),
        Int32(threadsPerBlock),
        Int32(numsPerBlock),
        Int32(numBlocks),
        Int32(numIterations),
        Int32(lastFFTLen),
        invnpbpruPowerTable,
        shmemSize,
        kernel1,
        invkernel2
    )

    return nttPlan, inttPlan
end

"""
    ntt!(vec::CuVector{T}, plan::NTTPlan{T}) where T<:Integer

Takes the NTT of `vec` using a plan generated from `plan_ntt`. Changes 
entries of `vec`, and doesn't return anything.
"""
function ntt!(vec::CuVector{T}, plan::NTTPlan{T}) where T<:Integer
    @assert length(vec) == plan.n

    twiddleStride = Int32(1)

    original = vec
    aux = CUDA.zeros(T, plan.n)

    for _ in 1:plan.numIterations - 1
        plan.kernel1(
            vec, plan.reducer, plan.npru, aux, plan.npbpruPowerTable, plan.numsPerBlock, twiddleStride;
            threads = plan.threadsPerBlock,
            blocks = plan.numBlocks,
            shmem = plan.shmemSize
        )

        twiddleStride *= plan.numsPerBlock

        temp = aux
        aux = vec
        vec = temp
    end

    plan.kernel2(
        vec, plan.reducer, aux, plan.npbpruPowerTable, plan.numsPerBlock, plan.lastFFTLen;
        threads = plan.threadsPerBlock,
        blocks = plan.numBlocks,
        shmem = plan.shmemSize
    )

    if pointer(aux) != pointer(original)
        original .= aux
    end

    # CUDA.unsafe_free!(aux)

    return nothing
end

"""
    intt!(vec::CuVector{T}, plan::INTTPlan{T}) where T<:Integer

Takes the normalized INTT of `vec` using a plan generated from `plan_ntt`. 
Changes entries of `vec`, and doesn't return anything. 
"""
function intt!(vec::CuVector{T}, plan::INTTPlan{T}) where T<:Integer
    @assert length(vec) == plan.n

    twiddleStride = Int32(1)

    original = vec
    aux = CUDA.zeros(T, plan.n)

    for _ in 1:plan.numIterations - 1
        plan.kernel1(
            vec, plan.reducer, plan.npru, aux, plan.npbpruPowerTable, plan.numsPerBlock, twiddleStride;
            threads = plan.threadsPerBlock,
            blocks = plan.numBlocks,
            shmem = plan.shmemSize
        )

        twiddleStride *= plan.numsPerBlock

        temp = aux
        aux = vec
        vec = temp
    end

    plan.kernel2(
        vec, plan.reducer, aux, plan.npbpruPowerTable, plan.lenInverse, plan.numsPerBlock, plan.lastFFTLen;
        threads = plan.threadsPerBlock,
        blocks = plan.numBlocks,
        shmem = plan.shmemSize
    )

    if pointer(aux) != pointer(original)
        original .= aux
    end

    # CUDA.unsafe_free!(aux)

    return nothing
end

function ntt_kernel1!(vec::CuDeviceVector{T}, p::Reducer{T}, npru::T, aux::CuDeviceVector{T}, npbpruPowerTable::CuDeviceVector{T}, numsPerBlock::Int32, twiddleStride::Int32) where T<:Integer
    @inbounds begin
    log2FFTLen = intlog2(numsPerBlock)
    
    # Load nums into shared memory
    shared = CuDynamicSharedArray(T, numsPerBlock)

    fftIdx = threadIdx().x - o # mov r2, tid
    globalIdx = blockIdx().x + fftIdx * gridDim().x 

    val = vec[globalIdx]
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[bitreversed + o] = val

    fftIdx += blockDim().x
    globalIdx += gridDim().x * blockDim().x

    val = vec[globalIdx]
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[bitreversed + o] = val

    # Load table into shared memory
    sharedTable = CuDynamicSharedArray(T, numsPerBlock, numsPerBlock * sizeof(T))

    idx = threadIdx().x

    sharedTable[idx] = npbpruPowerTable[idx]
    idx += blockDim().x
    sharedTable[idx] = npbpruPowerTable[idx]

    CUDA.sync_threads()

    # FFT in shared memory
    
    k = threadIdx().x - o
    for i in o:log2FFTLen
        m = o << i
        m2 = m >> o
        bits = log2FFTLen - i
        mask = (o << bits) - o
        
        idx1 = m * (k & mask) + (k >> bits) + o
        idx2 = idx1 + m2
        theta = sharedTable[((numsPerBlock >> i) * (k >> bits)) + o]
        # theta = npbpruPowerTable[((numsPerBlock >> i) * (k >> bits)) + o]
        t = mul_mod(theta, shared[idx2], p)
        u = shared[idx1]

        shared[idx1] = add_mod(u, t, p)
        shared[idx2] = sub_mod(u, t, p)

        CUDA.sync_threads()
    end

    # Apply twiddle and write to output array
    # Theoretically I could make this coalesced for 64-bit types as well
    blockTwiddleStride = ((blockIdx().x - o) ÷ twiddleStride) * twiddleStride
    
    fftIdx = threadIdx().x
    globalIdx = fftIdx + (blockIdx().x - o) * numsPerBlock

    shared[fftIdx] = mul_mod(shared[fftIdx], power_mod(npru, (fftIdx - o) * blockTwiddleStride, p), p)
    aux[globalIdx] = shared[fftIdx]

    globalIdx += blockDim().x
    fftIdx += blockDim().x

    shared[fftIdx] = mul_mod(shared[fftIdx], power_mod(npru, (fftIdx - o) * blockTwiddleStride, p), p)
    aux[globalIdx] = shared[fftIdx]
    end

    return nothing
end

function ntt_kernel2!(vec::CuDeviceVector{T}, p::Reducer{T}, aux::CuDeviceVector{T}, npbpruPowerTable::CuDeviceVector{T}, numsPerBlock::Int32, fftLen::Int32) where T<:Integer
    @inbounds begin

    shared = CuDynamicSharedArray(T, numsPerBlock)
    log2FFTLen = intlog2(fftLen)
    fftLenOver2 = fftLen >> o

    fftsPerBlock = numsPerBlock >> log2FFTLen

    # Load values into shmem
    virtualThreadIdx = threadIdx().x - o
    globalIdx = blockIdx().x + virtualThreadIdx * gridDim().x
    globalIdxStride = gridDim().x * blockDim().x
    
    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx >> log2FFTLen
    fftIdx = transposedIdx & ((o << log2FFTLen) - o)
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[fftNum * fftLen + bitreversed + o] = val

    virtualThreadIdx += blockDim().x
    globalIdx += globalIdxStride

    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx >> log2FFTLen
    fftIdx = transposedIdx & ((o << log2FFTLen) - o)
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[fftNum * fftLen + bitreversed + o] = val

    # Load table into shared memory
    sharedTable = CuDynamicSharedArray(T, numsPerBlock, numsPerBlock * sizeof(T))

    idx = threadIdx().x
    sharedTable[idx] = npbpruPowerTable[idx]
    idx += blockDim().x
    sharedTable[idx] = npbpruPowerTable[idx]

    CUDA.sync_threads()
    # FFT
    for i in o:log2FFTLen
        m = o << i
        m2 = m >> o
        bits = log2FFTLen - i
        mask = (o << bits) - o

        k = threadIdx().x - o
        gh_j = k % fftLenOver2
        fftNum = k ÷ fftLenOver2
        idx1 = m * (gh_j & mask) + (gh_j >> bits) + (fftLen * fftNum) + o
        idx2 = idx1 + m2
        theta = sharedTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + o]
        # theta = npbpruPowerTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + o]
        t = mul_mod(theta, shared[idx2], p)
        u = shared[idx1]

        shared[idx1] = add_mod(u, t, p)
        shared[idx2] = sub_mod(u, t, p)
        
        CUDA.sync_threads()
    end

    fftIdx = threadIdx().x - o
    globalIdx = fftIdx + (blockIdx().x - o) * numsPerBlock
    log2n = intlog2(length(vec))

    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + o] = shared[fftIdx + o]
    globalIdx += blockDim().x
    fftIdx += blockDim().x

    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + o] = shared[fftIdx + o]
    end
    return nothing
end

# Exact same as ntt_kernel2!, except multiplies everything by lenInverse at the end.
function ntt_kernel3!(vec::CuDeviceVector{T}, p::Reducer{T}, aux::CuDeviceVector{T}, npbpruPowerTable::CuDeviceVector{T}, lenInverse::T, numsPerBlock::Int32, fftLen::Int32) where T<:Integer
    @inbounds begin

    shared = CuDynamicSharedArray(T, numsPerBlock)
    log2FFTLen = intlog2(fftLen)
    fftLenOver2 = fftLen >> o

    fftsPerBlock = numsPerBlock >> log2FFTLen

    # Load values into shmem
    virtualThreadIdx = threadIdx().x - o
    globalIdx = blockIdx().x + virtualThreadIdx * gridDim().x
    globalIdxStride = gridDim().x * blockDim().x
    
    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx >> log2FFTLen
    fftIdx = transposedIdx & ((o << log2FFTLen) - o)
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[fftNum * fftLen + bitreversed + o] = val

    virtualThreadIdx += blockDim().x
    globalIdx += globalIdxStride

    val = vec[globalIdx]
    transposedIdx = get_transposed_index(virtualThreadIdx, fftsPerBlock, fftLen)
    fftNum = transposedIdx >> log2FFTLen
    fftIdx = transposedIdx & ((o << log2FFTLen) - o)
    bitreversed = bit_reverse(fftIdx, log2FFTLen)
    shared[fftNum * fftLen + bitreversed + o] = val

    # Load table into shared memory
    sharedTable = CuDynamicSharedArray(T, numsPerBlock, numsPerBlock * sizeof(T))

    idx = threadIdx().x
    sharedTable[idx] = npbpruPowerTable[idx]
    idx += blockDim().x
    sharedTable[idx] = npbpruPowerTable[idx]

    CUDA.sync_threads()
    # FFT
    for i in o:log2FFTLen
        m = o << i
        m2 = m >> o
        bits = log2FFTLen - i
        mask = (o << bits) - o

        k = threadIdx().x - o
        gh_j = k % fftLenOver2
        fftNum = k ÷ fftLenOver2
        idx1 = m * (gh_j & mask) + (gh_j >> bits) + (fftLen * fftNum) + o
        idx2 = idx1 + m2
        theta = sharedTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + o]
        # theta = npbpruPowerTable[((fftLen >> i) * (fftsPerBlock) * (gh_j >> bits)) + o]
        t = mul_mod(theta, shared[idx2], p)
        u = shared[idx1]

        shared[idx1] = add_mod(u, t, p)
        shared[idx2] = sub_mod(u, t, p)
        
        CUDA.sync_threads()
    end

    fftIdx = threadIdx().x - o
    globalIdx = fftIdx + (blockIdx().x - o) * numsPerBlock
    log2n = intlog2(length(vec))

    shared[fftIdx + o] = mul_mod(shared[fftIdx + o], lenInverse, p)
    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + o] = shared[fftIdx + o]
    globalIdx += blockDim().x
    fftIdx += blockDim().x

    shared[fftIdx + o] = mul_mod(shared[fftIdx + o], lenInverse, p)
    aux[final_transpose(globalIdx, log2n, numsPerBlock, fftLen) + o] = shared[fftIdx + o]
    end
    return nothing
end