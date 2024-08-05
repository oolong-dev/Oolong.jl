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

function launch(f; p=8)
    if parse(Int, ENV["RANK"]) == 0
        oolong_master()
    end

    # https://github.com/JuliaLang/Distributed.jl/blob/6a0383b9daf5d7f364fd6fc580aac975ca759edd/src/managers.jl#L475
    env = Dict{String,String}()
    project = Base.ACTIVE_PROJECT[]
    if project !== nothing && get(env, "JULIA_PROJECT", nothing) === nothing
        env["JULIA_PROJECT"] = project
    end

    dir = "$(pwd())"

    local_workers = Base.Process[]

    for i in 1:p
        env["LOCAL_RANK"] = string(i - 1)
        cmd = `julia -e "using Oolong; Oolong.join_cluster()"`
        ps = open(pipeline(detach(setenv(addenv(cmd, env), dir=dir)); stdout=stdout, stderr=stderr), "r")
        push!(local_workers, ps)
    end
    success(local_workers)
end