using Distributed
using NCCL
using CUDA

struct ProcessGroup
    rank::Int
    ranks::Vector{Int}
    communicator::Communicator
end

const DEFAULT_PROCESS_GROUP = Ref{ProcessGroup}()

function init_process_group()
    device!(parse(Int, ENV["LOCAL_RANK"]))
    if myid() == 1
        comm_id = NCCL.UniqueID()
    end
    DEFAULT_PROCESS_GROUP[] = ProcessGroup()
end

get_rank(gp::ProcessGroup) = nothing