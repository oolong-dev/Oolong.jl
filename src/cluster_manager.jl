# This file is inspired by ElasticManager from ClusterManagers.jl

using Distributed
using Sockets

function oolong_master()
end

function oolong_worker()
    master_addr = ENV["MASTER_ADDR"]
    master_port = parse(Int, ENV["MASTER_PORT"])
    rank = parse(Int, ENV["RANK"])
    local_rank = parse(Int, ENV["LOCAL_RANK"])
    _cookie = get(ENV, "OOLONG_COOKIE", "oolong")
    cookie = "$(rank):$(local_rank):$(_cookie)"

    # TODO: worker_timeout()
    c = connect(master_addr, master_port)

    write(c, rpad(cookie, HDR_COOKIE_LEN)[1:HDR_COOKIE_LEN])
    start_worker(c, cookie)
end