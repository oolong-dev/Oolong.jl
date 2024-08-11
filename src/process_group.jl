export get_rank

using Distributed
using NCCL
using CUDA

struct ProcessGroup
    ranks::Vector{Int}
    communicator::Communicator
end

const DEFAULT_PROCESS_GROUP = Ref{ProcessGroup}()

function init_process_group(id)
    world_size = parse(Int, ENV["WORLD_SIZE"])
    rank = parse(Int, ENV["RANK"])
    local_rank = parse(Int, ENV["LOCAL_RANK"])
    nproc_per_node = parse(Int, ENV["NPROC_PER_NODE"])

    c = Communicator(world_size * nproc_per_node, rank * nproc_per_node + local_rank; unique_id=id)
    DEFAULT_PROCESS_GROUP[] = ProcessGroup(0:nproc_per_node*world_size-1 |> collect, c)
end

get_rank() = get_rank(DEFAULT_PROCESS_GROUP[])
get_rank(pg::ProcessGroup) = NCCL.rank(pg.communicator)