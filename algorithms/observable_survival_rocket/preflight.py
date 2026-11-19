"""Read-only GPU/dependency preflight. Does not install or start workloads."""
import importlib.util
import json
import platform
import sys

def main():
    import torch
    deps={name:importlib.util.find_spec(name) is not None for name in
          ('genesis','rsl_rl','tensordict','numpy')}
    gpu=torch.cuda.is_available()
    info={'python':platform.python_version(),'torch':torch.__version__,
          'cuda_available':gpu,'dependencies':deps,'training_started':False}
    if gpu:
        free,total=torch.cuda.mem_get_info()
        info.update(gpu_name=torch.cuda.get_device_name(),free_gib=free/2**30,total_gib=total/2**30)
    print(json.dumps(info,indent=2))
    return 0 if gpu and all(deps.values()) else 2

if __name__=='__main__':
    sys.exit(main())
