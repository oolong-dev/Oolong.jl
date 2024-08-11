# This file is inspired by ElasticManager from ClusterManagers.jl

using Distributed
using Sockets
using CUDA

const DEFAULT_COOKIE = string(@__MODULE__)
const HDR_COOKIE_LEN = Distributed.HDR_COOKIE_LEN
const RANK_LEN = 4
const LOCAL_RANK_LEN = 2

struct OolongManager <: ClusterManager
    workers::Matrix{WorkerConfig}

    function OolongManager(; addr=IPv4("127.0.0.1"), port=9009, nproc_per_node=ndevices(), nnode=1, cookie=DEFAULT_COOKIE)
        @assert ':' ∉ cookie
        Distributed.init_multi()
        cluster_cookie(cookie) # it's set to rand string in `init_multi`, we should reset it here
        l_sock = listen(addr, port)

        m = new(Matrix{WorkerConfig}(undef, nproc_per_node, nnode))

        for i in 1:nnode, j in 1:nproc_per_node
            s = accept(l_sock)
            cookie = read(s, HDR_COOKIE_LEN) |> String
            @assert cookie == cluster_cookie()
            rank = parse(Int, read(s, RANK_LEN) |> String |> strip)
            local_rank = parse(Int, read(s, LOCAL_RANK_LEN) |> String |> strip)
            addprocs(m; rank, local_rank, topology="master_worker", io=s)
        end
        m
    end
end

function Distributed.launch(m::OolongManager, params::Dict, launched::Array, c::Condition)
    # The workers have already been started.
    wc = WorkerConfig()
    wc.io = params[:io]
    wc.userdata = Dict{String,Any}("rank" => params[:rank], "local_rank" => params[:local_rank])
    push!(launched, wc)
    notify(c)
end

function Distributed.manage(m::OolongManager, id::Integer, config::WorkerConfig, op::Symbol)
    config.userdata["pid"] = id
    rank = config.userdata["rank"]
    local_rank = config.userdata["local_rank"]

    if op == :register
        m.workers[local_rank+1, rank+1] = config
    elseif op == :deregister
        @warn "Worker $rank:$local_rank has exited"
    end
end

function oolong_worker(; addr=IPv4("127.0.0.1"), port=9009, rank=0, local_rank=0, cookie=DEFAULT_COOKIE)
    cookie = rpad(cookie, HDR_COOKIE_LEN)[1:HDR_COOKIE_LEN]
    # TODO: worker_timeout()
    c = connect(addr, port)
    write(c, cookie)
    write(c, lpad(rank, RANK_LEN))
    write(c, lpad(local_rank, LOCAL_RANK_LEN))
    start_worker(c, cookie)
end