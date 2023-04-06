import os
import jax
from pprint import pprint

procid = int(os.getenv('SLURM_PROCID'))
n_local_gpus = int(os.getenv('SLURM_GPUS_PER_TASK'))
print(f'\n ## process id: {procid}')

jax.distributed.initialize(local_device_ids=[i for i in range(n_local_gpus)])

print(f'local device count: {jax.local_device_count()}')
pprint(jax.local_devices())

if procid == 0:
    print(f'global device count: {jax.device_count()}')
    pprint(jax.devices())

xs = jax.numpy.ones(jax.local_device_count())
print(jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs))
