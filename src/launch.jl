export launch

#=
Design

1. Assume users have a workable cluster which can successfully run a pytorch distributed job.

  - Usually this means that the followiing environment varialbes are set:
    - MASTER_ADDR
    - MASTER_PORT
    - WORLD_SIZE
    - RANK

2. The entrypoint for Oolong is close to torchrun. On each node, by calling `julia -e "using Oolong; launch(;nproc_per_node=8)" main.jl`, we first spawn the specified `nproc_per_node` local processes. Note that these processes are all *worker*s. And the *master* process is the launcher on RANK 0.

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