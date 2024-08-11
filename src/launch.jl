export launch

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
    nnode=get(ENV, "WORLD_SIZE", "1"),
    rank=get(ENV, "RANK", "0"),
    master_addr=get(ENV, "MASTER_ADDR", "localhost"),
    master_port=get(ENV, "MASTER_PORT", "9002")
)
    if parse(Int, nnode) == 1
        addprocs(nproc_per_node)
    end

    # 1. Setup the header
    # 2. Spawn workers
    local_workers = Base.Process[]
    # https://github.com/JuliaLang/Distributed.jl/blob/6a0383b9daf5d7f364fd6fc580aac975ca759edd/src/managers.jl#L475
    env = Dict{String,String}()
    project = Base.ACTIVE_PROJECT[]
    if project !== nothing && get(env, "JULIA_PROJECT", nothing) === nothing
        env["JULIA_PROJECT"] = project
    end

    dir = "$(pwd())"

    for i in 1:p
        env["LOCAL_RANK"] = string(i - 1)
        cmd = `julia -e "using Oolong; Oolong.join_cluster()"`
        ps = open(pipeline(detach(setenv(addenv(cmd, env), dir=dir)); stdout=stdout, stderr=stderr), "r")
        push!(local_workers, ps)
    end
    success(local_workers)
end