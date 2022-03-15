from gym.envs.registration import register

register(
    id='grn-control-v0',
    entry_point='gym_grn_control.envs:GRNControlEnv',
)
register(
    id='grn-control-simple-v0',
    entry_point='gym_grn_control.envs:GRNControlSimpleEnv',
)
