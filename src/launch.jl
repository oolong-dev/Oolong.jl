#=
# TL;DR

Single Node, Multi GPU

`oolong main.jl --nproc_per_node 8`

Multi Node, Multi GPU

```bash
export MASTER_ADDR=10.1.2.3
export MASTER_PORT=9002
export WORLD_SIZE=16
export RANK=0
oolong main.jl --nproc_per_node 8
```

`oolong` => `julia --project -e "using Oolong; launch()"`

# Design

1. Assume users have a workable cluster which can successfully run a pytorch distributed job.

  - Usually this means that the followiing environment varialbes are set:
    - MASTER_ADDR
    - MASTER_PORT
    - WORLD_SIZE
    - RANK

2. The entrypoint for Oolong is close to torchrun. In single node mode, we simply setup `nproc_per_node` local workers (just the same as `julia -p` ). In the distributed mode, each worker will try to connect the the main process specified at `MASTER_ADDR` and `MASTER_PORT`. The implementation is similiar to `ElasticManager` from `ClusterManagers.jl`, except that we need to wait until all the workers are connected.

3. Once all the workers joined the cluster, the master process will do a remote call and setup the default process group. After that, we'll execute `include("main.jl")` everywhere.
=#

using Distributed
using CUDA

function launch(f;
    nproc_per_node=ndevices(),
    nnode=parse(Int, get(ENV, "WORLD_SIZE", "1")),
    rank=parse(Int, get(ENV, "RANK", "0")),
    master_addr=get(ENV, "MASTER_ADDR", DEFAULT_MASTER_ADDR),
    master_port=parse(Int, get(ENV, "MASTER_PORT", "$DEFAULT_MASTER_PORT")),
    cookie=get(ENV, "OOLONG_COOKIE", DEFAULT_COOKIE),
)
    if rank == 0
        @info "preparing oolong manager..."
        m = OolongManager(; addr=master_addr, port=master_port, nproc_per_node=nproc_per_node, nnode=nnode, cookie=cookie)
    end

    @info "setting up local workers..."
    local_workers = Base.Process[]
    # https://github.com/JuliaLang/Distributed.jl/blob/6a0383b9daf5d7f364fd6fc580aac975ca759edd/src/managers.jl#L475
    env = Dict{String,String}(
        "MASTER_ADDR" => master_addr,
        "MASTER_PORT" => string(master_port),
        "WORLD_SIZE" => string(nnode),
        "NPROC_PER_NODE" => string(nproc_per_node),
        "RANK" => string(rank),
        "OOLONG_COOKIE" => cookie,
    )
    project = Base.ACTIVE_PROJECT[]
    if project !== nothing && get(env, "JULIA_PROJECT", nothing) === nothing
        env["JULIA_PROJECT"] = project
    end
    dir = "$(pwd())"
    for i in 1:nproc_per_node
        env["LOCAL_RANK"] = string(i - 1)
        cmd = `julia -e "using Oolong; Oolong.join_cluster()"`
        ps = open(pipeline(detach(setenv(addenv(cmd, env), dir=dir)); stdout=stdout, stderr=stderr), "r")
        push!(local_workers, ps)
    end

    if rank == 0
        init(m)
        @everywhere workers() include($f)
    end
    success(local_workers)
end

function init(m::OolongManager)
    @info "waiting for all workers to join..."
    wait(m.taskref[])
    @info "all workers joined!"

    @eval @everywhere begin
        using CUDA
        using NCCL
    end

    # 1. setup device to LOCAL_RANK
    @info "setting device..."
    @everywhere workers() begin
        CUDA.device!(parse(Int, ENV["LOCAL_RANK"]))
    end

    # 2. create UID at rank 0
    @info "creating NCCL UniqueID at rank 0..."
    uid = remotecall_fetch(m.workers[1].userdata["pid"]) do
        NCCL.UniqueID()
    end

    # 3. init NCCL process group
    @info "initializing NCCL communicator everywhere..."
    @everywhere workers() begin
        Oolong.init_process_group($uid)
    end
    @info "all done! ready to execute user provided script."
end