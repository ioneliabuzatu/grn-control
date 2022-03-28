from gym.envs.registration import register

from gym_gene_control.envs.grn_control import GRNControlEnv
from gym_gene_control.envs.grn_control_simple import GRNControlSimpleEnv

register(
    id='gene_control-v0',
    entry_point='gym_gene_control:GRNControlEnv',
)
register(
    id='gene_control-simple-v0',
    entry_point='gym_gene_control:GRNControlSimpleEnv',
)
