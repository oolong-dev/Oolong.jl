# Hello World!

Use this script to make sure that your environment and workflow works with Oolong.jl

## Single Node Multiple Devices

```bash
julia --project -e "using Oolong; Oolong.launch(\"demo/hello_world/hello_world.jl\")"
```

## Multi Nodes Multiple Devices

In multi-node mode, the main command to execute is just the same with the single node mode. The main difference is that we need to setup some extra environment variables. Usually these variables are set by the job scheduler (like [Volcano](https://volcano.sh/) in K8S).

```bash
export MASTER_ADDR=YOUR_ACCESSIBLE_ADDRESS
export MASTER_PORT=YOUR_AVAILABLE_PORT
export WORLD_SIZE=2
export RANK=0
julia --project -e "using Oolong; Oolong.launch(\"demo/hello_world/hello_world.jl\")"
```

If you are coming from the PyTorch world, you may find the above environment variables are very similar to the ones used in PyTorch's distributed training.

## Single Node Multiple Devices (Simulate Multi Nodes)

In this mode, we simulate a multi-node environment on a single node. This is useful for debugging and testing.

Process 1:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9009
export WORLD_SIZE=2
export RANK=0
CUDA_VISIBLE_DEVICES=0,1,2,3 julia --project -e "using Oolong; Oolong.launch(\"demo/hello_world/hello_world.jl\")"
```


Process 2:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9009
export WORLD_SIZE=2
export RANK=1
CUDA_VISIBLE_DEVICES=4,5,6,7 julia --project -e "using Oolong; Oolong.launch(\"demo/hello_world/hello_world.jl\")"
```
