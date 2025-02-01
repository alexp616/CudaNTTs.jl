global const o = Int32(1)

@inline function CTUnit(U::Ref{T}, V::Ref{T}, root::T, m::Reducer{T})::Nothing where T<:Unsigned
    u_ = U[]
    v_ = mul_mod(V[], root, m)

    U[] = add_mod(u_, v_, m)
    V[] = sub_mod(u_, v_, m)

    return nothing
end

@inline function GSUnit(U::Ref{T}, V::Ref{T}, root::T, m::Reducer{T})::Nothing where T<:Unsigned
    u_ = U[]
    v_ = mul_mod(V[], root, m)

    U[] = add_mod(u_, v_, m)
    V[] = sub_mod(u_, v_, m)

    return nothing
end

function ntt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:Unsigned

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o
    block_z = blockIdx().z - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - logm - o)
    t_ = shared_index
    m = o << logm

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset) + 
                     (block_z << N_power)

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

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

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
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, not_last_kernel::Bool)::Nothing where T<:Unsigned

    @inbounds begin

    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o
    block_z = blockIdx().z - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - logm - o)
    t_ = shared_index
    m = o << logm

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset) + 
                     (block_z << N_power)

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

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    else
        for _ in o:(shared_index - Int32(5))
            CUDA.sync_threads()

            current_root_index = (omega_address >> t_2)

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()

        for _ in o:Int32(6)
            current_root_index = (omega_address >> t_2)

            CTUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), root_of_unity_table[current_root_index + o], modulus)

            t = t >> 1
            t_2 -= o
            t_ -= o
            m <<= 1

            in_shared_address = ((shared_address >> t_) << t_) + shared_address
        end
        CUDA.sync_threads()
    end

    polynomial_out[global_address + o] = shared_memory[shared_address + o]
    polynomial_out[global_address + offset + o] = shared_memory[shared_address + blockDim().x * blockDim().y + o]
    
    end # inbounds
    return nothing
end

function intt_kernel1!(polynomial_in::CuDeviceVector{T}, polynomial_out::CuDeviceVector{T}, 
    inverse_root_of_unity_table::CuDeviceVector{T}, modulus::Reducer{T}, shared_index::Int32, logm::Int32, k::Int32, 
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:Unsigned

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o
    block_z = blockIdx().z - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - k - o)
    t_ = shared_index + o - outer_iteration_count
    m = o << logm

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_x) + (Int32(2) * block_y * offset) + 
                     (block_z << N_power)
    
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

        GSUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), inverse_root_of_unity_table[current_root_index + o], modulus)

        t = t << 1
        t_2 += o
        t_ += o
        m >>= 1

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
    outer_iteration_count::Int32, N_power::Int32, shmem_length::Int32, n_inverse::T, last_kernel::Bool)::Nothing where T<:Unsigned

    @inbounds begin
    idx_x = threadIdx().x - o
    idx_y = threadIdx().y - o
    block_x = blockIdx().x - o
    block_y = blockIdx().y - o
    block_z = blockIdx().z - o

    shared_memory = CuDynamicSharedArray(T, shmem_length)

    t_2 = N_power - logm - o
    offset = o << (N_power - k - o)
    t_ = shared_index + 1 - outer_iteration_count
    m = o << logm

    global_address = idx_x + (idx_y * (offset ÷ (o << (outer_iteration_count - o)))) + 
                     (blockDim().x * block_y) + (Int32(2) * block_x * offset) + 
                     (block_z << N_power)
    
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

        GSUnit(Ref(shared_memory, in_shared_address + o), Ref(shared_memory, in_shared_address + t + o), inverse_root_of_unity_table[current_root_index + o], modulus)

        t = t << 1
        t_2 += o
        t_ += o
        m >>= 1

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


