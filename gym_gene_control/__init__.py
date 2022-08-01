from gym.envs.registration import register

from gym_gene_control.envs.grn_control import GRNControlEnvThesis
from gym_gene_control.envs.grn_control import GRNControlEnvDS4
from gym_gene_control.envs.grn_control_simple import GRNControlSimpleEnv

register(
    id='gene_control-v1',
    entry_point='gym_gene_control:GRNControlEnvThesis',
)

register(
    id='gene_control-v0',
    entry_point='gym_gene_control:GRNControlEnvDS4',
)

register(
    id='gene_control-simple-v0',
    entry_point='gym_gene_control:GRNControlSimpleEnv',
)