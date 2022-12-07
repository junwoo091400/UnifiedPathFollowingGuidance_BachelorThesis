from gym.envs.registration import register
register(
    id='fixedwing-longitudinal', 
    entry_point='windywings.envs:FWLongitudinal'
)

register(
    id='fixedwing-lateral', 
    entry_point='windywings.envs:FWLateral'
)

register(
    id='fixedwing-lateral-npfg', 
    entry_point='windywings.envs:FWLateralNPFG'
)