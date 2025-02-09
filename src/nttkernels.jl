@inline function CTUnit(U::Core.LLVMPtr{T}, V::Core.LLVMPtr{T}, root::T, m::Reducer{T})::Nothing where T<:INTTYPES
    u_ = unsafe_load(U)
    v_ = mul_mod(unsafe_load(V), root, m)

    unsafe_store!(U, add_mod(u_, v_, m))
    unsafe_store!(V, sub_mod(u_, v_, m))

    return nothing
end

@inline function GSUnit(U::Core.LLVMPtr{T}, V::Core.LLVMPtr{T}, root::T, m::Reducer{T})::Nothing where T<:INTTYPES
    u_ = unsafe_load(U)
    v_ = unsafe_load(V)

    unsafe_store!(U, add_mod(u_, v_, m))

    v_ = sub_mod(u_, v_, m)
    unsafe_store!(V, mul_mod(v_, root, m))

    return nothing
end

function small_ntt_kernel!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, N_power::Int32) where T<:INTTYPES
    @inbounds begin
    idx_x = threadIdx().x - o
    
    shared_memory = CuDynamicSharedArray(T, length(polynomial_in))
    shared_memory[threadIdx().x] = polynomial_in[threadIdx().x]
    shared_memory[threadIdx().x + blockDim().x] = polynomial_in[threadIdx().x + blockDim().x]

    t_ = N_power - o
    t = blockDim().x

    in_shared_address = ((idx_x >> t_) << t_) + idx_x
    current_root_index = zero(Int32)

    for _ in o:N_power
        CUDA.sync_threads()
        current_root_index = idx_x >> t_

        CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)
        t >>= 1
        t_ -= o

        in_shared_address = (((threadIdx().x - o) >> t_) << t_) + threadIdx().x - o
    end
    CUDA.sync_threads()

    polynomial_out[threadIdx().x] = shared_memory[threadIdx().x]
    polynomial_out[threadIdx().x + blockDim().x] = shared_memory[threadIdx().x + blockDim().x]
    end

    return
end

function ntt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - logm - o)
    t_ = shared_index

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset)

    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                    (blockDim().x * block_x) + (block_y * offset)
    
    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    if (not_last_kernel)
        for _ in o:outer_iteration_count
            CUDA.sync_threads()
        
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    end

    polynomial_out[global_address + o] = shared_memory[shared_address + o]
    polynomial_out[global_address + offset + o] = shared_memory[shared_address + blockDim().x * blockDim().y + o]

    end # inbounds
    return nothing
end

function ntt_kernel2!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - logm - o)
    t_ = shared_index

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset)

    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                    (blockDim().x * block_y) + (block_x * offset)
    
    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    if (not_last_kernel)
        for _ in o:outer_iteration_count
            CUDA.sync_threads()
        
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)
            
            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    end

    polynomial_out[global_address + o] = shared_memory[shared_address + o]
    polynomial_out[global_address + offset + o] = shared_memory[shared_address + blockDim().x * blockDim().y + o]
    
    end # inbounds
    return nothing
end

function small_intt_kernel!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, N_power::Int32, n_inverse::T) where T<:INTTYPES

    idx_x = threadIdx().x - o

    shared_memory = CuDynamicSharedArray(T, length(polynomial_in))

    t_ = Int32(0)

    shared_memory[threadIdx().x] = polynomial_in[threadIdx().x]
    shared_memory[threadIdx().x + blockDim().x] = polynomial_in[threadIdx().x + blockDim().x]

    t = o << t_
    in_shared_address = ((idx_x >> t_) << t_) + idx_x
    current_root_index = zero(Int32)

    for _ in o:N_power
        CUDA.sync_threads()

        current_root_index = idx_x >> t_

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), inverse_root_of_unity_table[current_root_index + o], modulus)

        t = t << 1
        t_ += o

        in_shared_address = ((idx_x >> t_) << t_) + idx_x
    end
    CUDA.sync_threads()

    polynomial_out[threadIdx().x] = mul_mod(shared_memory[threadIdx().x], n_inverse, modulus)
    polynomial_out[threadIdx().x + blockDim().x] = mul_mod(shared_memory[threadIdx().x + blockDim().x], n_inverse, modulus)
    return nothing
end

function intt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, k::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - k - o)
    t_ = shared_index + o - outer_iteration_count

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset)
    
    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (block_y * offset)

    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    for _ in o:outer_iteration_count
        CUDA.sync_threads()
        
        current_root_index = (omega_address >> t_2)

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), inverse_root_of_unity_table[current_root_index + o], modulus)

        t = t << 1
        t_2 += o
        t_ += o

        in_shared_address = ((shared_address >> t_) << t_) + shared_address
    end
    CUDA.sync_threads()

    if (last_kernel)
        polynomial_out[global_address + o] = mul_mod(shared_memory[shared_address + o], n_inverse, modulus)
        polynomial_out[global_address + offset + o] = mul_mod(shared_memory[shared_address + (blockDim().x * blockDim().y) + o], n_inverse, modulus)
    else
        polynomial_out[global_address + o] = shared_memory[shared_address + o];
        polynomial_out[global_address + offset + o] = shared_memory[shared_address + (blockDim().x * blockDim().y) + o];
    end

    end
    return nothing
end

function intt_kernel2!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, k::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - k - o)
    t_ = shared_index + 1 - outer_iteration_count

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset)
    
    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (block_x * offset)

    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    for _ in o:outer_iteration_count
        CUDA.sync_threads()
        
        current_root_index = (omega_address >> t_2)

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), inverse_root_of_unity_table[current_root_index + o], modulus)

        t = t << 1
        t_2 += o
        t_ += o

        in_shared_address = ((shared_address >> t_) << t_) + shared_address
    end
    CUDA.sync_threads()
    if (last_kernel)
        polynomial_out[global_address + o] = mul_mod(shared_memory[shared_address + o], n_inverse, modulus)
        polynomial_out[global_address + offset + o] = mul_mod(shared_memory[shared_address + blockDim().x * blockDim().y + o], n_inverse, modulus)
    else
        polynomial_out[global_address + o] = shared_memory[shared_address + o];
        polynomial_out[global_address + offset + o] = shared_memory[shared_address + (blockDim().x * blockDim().y) + o];
    end
    
    end
    return nothing
end

# MEMORYEFFICIENT IMPLEMENTATIONS
# Probably don't need basically a copy of all of the code above, but this is the
# easy solution
#
#
#
#
#
#
#

# memory efficient small ntt kernel
function me_small_ntt_kernel!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, root_of_unity::T, modulus::Reducer{T}, N_power::Int32) where T<:INTTYPES
    @inbounds begin
    idx_x = threadIdx().x - o
    
    shared_memory = CuDynamicSharedArray(T, length(polynomial_in))
    shared_memory[threadIdx().x] = polynomial_in[threadIdx().x]
    shared_memory[threadIdx().x + blockDim().x] = polynomial_in[threadIdx().x + blockDim().x]
    
    # log2(n) minus 1
    log2nm1 = N_power - o
    t_ = log2nm1
    t = blockDim().x

    in_shared_address = ((idx_x >> t_) << t_) + idx_x
    current_root_index = zero(Int32)

    for _ in o:N_power
        CUDA.sync_threads()
        current_root_index = idx_x >> t_

        CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

        t >>= 1
        t_ -= o

        in_shared_address = (((threadIdx().x - o) >> t_) << t_) + threadIdx().x - o
    end
    CUDA.sync_threads()

    polynomial_out[threadIdx().x] = shared_memory[threadIdx().x]
    polynomial_out[threadIdx().x + blockDim().x] = shared_memory[threadIdx().x + blockDim().x]
    end

    return
end

function me_ntt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    root_of_unity::T, modulus::Reducer{T}, shared_index::Int32, logm::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    log2nm1 = N_power - o
    t_2 = log2nm1 - logm
    offset = o << t_2
    t_ = shared_index

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset)

    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                    (blockDim().x * block_x) + (block_y * offset)
    
    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    if (not_last_kernel)
        for _ in o:outer_iteration_count
            CUDA.sync_threads()
        
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    end

    polynomial_out[global_address + o] = shared_memory[shared_address + o]
    polynomial_out[global_address + offset + o] = shared_memory[shared_address + blockDim().x * blockDim().y + o]

    end # inbounds
    return nothing
end

function me_ntt_kernel2!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    root_of_unity::T, modulus::Reducer{T}, shared_index::Int32, logm::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    log2nm1 = N_power - o
    t_2 = log2nm1 - logm
    offset = o << t_2
    t_ = shared_index

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset)

    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                    (blockDim().x * block_y) + (block_x * offset)
    
    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    if (not_last_kernel)
        for _ in o:outer_iteration_count
            CUDA.sync_threads()
        
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(root_of_unity, current_root_index, log2nm1, modulus), modulus)
            
            t = t >> 1
            t_2 -= o
            t_ -= o

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    end

    polynomial_out[global_address + o] = shared_memory[shared_address + o]
    polynomial_out[global_address + offset + o] = shared_memory[shared_address + blockDim().x * blockDim().y + o]
    
    end # inbounds
    return nothing
end

function me_small_intt_kernel!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity::T, modulus::Reducer{T}, N_power::Int32, n_inverse::T) where T<:INTTYPES

    idx_x = threadIdx().x - o

    shared_memory = CuDynamicSharedArray(T, length(polynomial_in))

    t_ = Int32(0)

    shared_memory[threadIdx().x] = polynomial_in[threadIdx().x]
    shared_memory[threadIdx().x + blockDim().x] = polynomial_in[threadIdx().x + blockDim().x]

    log2nm1 = N_power - o
    t = o << t_
    in_shared_address = ((idx_x >> t_) << t_) + idx_x
    current_root_index = zero(Int32)

    for _ in o:N_power
        CUDA.sync_threads()

        current_root_index = idx_x >> t_

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(inverse_root_of_unity, current_root_index, log2nm1, modulus), modulus)

        t = t << 1
        t_ += o

        in_shared_address = ((idx_x >> t_) << t_) + idx_x
    end
    CUDA.sync_threads()

    polynomial_out[threadIdx().x] = mul_mod(shared_memory[threadIdx().x], n_inverse, modulus)
    polynomial_out[threadIdx().x + blockDim().x] = mul_mod(shared_memory[threadIdx().x + blockDim().x], n_inverse, modulus)
    return nothing
end

function me_intt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity::T, modulus::Reducer{T}, shared_index::Int32, logm::Int32, k::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    log2nm1 = N_power - o
    t_2 = log2nm1 - logm
    offset = o << (log2nm1 - k)
    t_ = shared_index + o - outer_iteration_count

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset)
    
    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (block_y * offset)

    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    for _ in o:outer_iteration_count
        CUDA.sync_threads()
        
        current_root_index = (omega_address >> t_2)

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(inverse_root_of_unity, current_root_index, log2nm1, modulus), modulus)

        t = t << 1
        t_2 += o
        t_ += o

        in_shared_address = ((shared_address >> t_) << t_) + shared_address
    end
    CUDA.sync_threads()

    if (last_kernel)
        polynomial_out[global_address + o] = mul_mod(shared_memory[shared_address + o], n_inverse, modulus)
        polynomial_out[global_address + offset + o] = mul_mod(shared_memory[shared_address + (blockDim().x * blockDim().y) + o], n_inverse, modulus)
    else
        polynomial_out[global_address + o] = shared_memory[shared_address + o];
        polynomial_out[global_address + offset + o] = shared_memory[shared_address + (blockDim().x * blockDim().y) + o];
    end

    end
    return nothing
end

function me_intt_kernel2!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity::T, modulus::Reducer{T}, shared_index::Int32, logm::Int32, k::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:INTTYPES

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    log2nm1 = N_power - o
    t_2 = log2nm1 - logm
    offset = o << (log2nm1 - k)
    t_ = shared_index + o - outer_iteration_count

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset)
    
    omega_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (block_x * offset)

    shared_address = idx_x + (idx_y * blockDim().x)

    shared_memory[shared_address + o] = polynomial_in[global_address + o]
    shared_memory[shared_address + (blockDim().x * blockDim().y) + o] = polynomial_in[global_address + offset + o]

    t = o << t_
    in_shared_address = ((shared_address >> t_) << t_) + shared_address
    current_root_index = zero(Int32)

    for _ in o:outer_iteration_count
        CUDA.sync_threads()
        
        current_root_index = (omega_address >> t_2)

        GSUnit(pointer(shared_memory, in_shared_address + o), pointer(shared_memory, in_shared_address + t + o), br_power_mod(inverse_root_of_unity, current_root_index, log2nm1, modulus), modulus)

        t = t << 1
        t_2 += o
        t_ += o

        in_shared_address = ((shared_address >> t_) << t_) + shared_address
    end
    CUDA.sync_threads()
    if (last_kernel)
        polynomial_out[global_address + o] = mul_mod(shared_memory[shared_address + o], n_inverse, modulus)
        polynomial_out[global_address + offset + o] = mul_mod(shared_memory[shared_address + blockDim().x * blockDim().y + o], n_inverse, modulus)
    else
        polynomial_out[global_address + o] = shared_memory[shared_address + o];
        polynomial_out[global_address + offset + o] = shared_memory[shared_address + (blockDim().x * blockDim().y) + o];
    end
    
    end
    return nothing
end