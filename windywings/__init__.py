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
    id='multicopter-fixedwing-lateral-npfg', 
    entry_point='windywings.envs:FWMCLateralNPFG'
)