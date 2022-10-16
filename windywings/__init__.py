from gym.envs.registration import register
register(
    id='fixedwing-longitudinal', 
    entry_point='windywings.envs:FWLongitudinal'
)
