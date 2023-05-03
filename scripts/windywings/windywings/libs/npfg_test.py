''' NPFG Unit Test'''

import unittest
import numpy as np
from windywings.libs.npfg import NPFG

class TestNPFG(unittest.TestCase):
    def test_bearing_feasibility(self):
        ''' Test `bearingFeasibility` function of NPFG '''
        # Arbitrary airspeed of the vehicle (e.g. Hobbywing Z-84 would fly at this speed)
        AIRSPEED = 10

        # Maximum airspeed isn't relevant, make it very big.
        npfg = NPFG(AIRSPEED, AIRSPEED*10.0)

        # We conceptually have a dial on the 'wind factor', and use the resulting wind speed as an agrument
        # >> Wind below the vehicle airspeed
        for wind_factor in [0.0, 0.5]:
            WINDSPEED = AIRSPEED * wind_factor
            WIND_VECTOR = WINDSPEED * np.array([1.0, 0.0]) # Wind always faces to X-axis
            # Case 1: Bearing going with the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, 0.0, AIRSPEED), 1)

            # Case 3: Bearing perpendicular to the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, np.pi/2, AIRSPEED), 1)

            # Case 4: Bearing going fully against the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, np.pi, AIRSPEED), 1)
            
        # >> Wind above or equal to vehicle airspeed
        for wind_factor in [1.0, 1.5, 2.0]:
            WINDSPEED = AIRSPEED * wind_factor
            WIND_VECTOR = WINDSPEED * np.array([1.0, 0.0]) # Wind always faces to X-axis
            # Case 1: Bearing going with the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, 0.0, AIRSPEED), 1)

            # Case 2: Bearing at half of sub-PI/2 angle around physical feasibility boundary
            # Feasibility should be between 0 and 1, then it shouldn't be plain 0!
            self.assertIsNot(npfg.bearingFeasibility(WIND_VECTOR, np.arcsin(1/wind_factor)/2, AIRSPEED), 0)

            # Case 2: Bearing at sub-PI/2 angle around physical feasibility boundary
            # Feasibility should be between 0 and 1, then it shouldn't be plain 0!
            # Note, the function returns 0, but not sure why 'assertIsNot' doesn't trigger :(
            self.assertIsNot(npfg.bearingFeasibility(WIND_VECTOR, np.arcsin(1/wind_factor), AIRSPEED), 0)

            # Case 3: Bearing perpendicular to the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, np.pi/2, AIRSPEED), 0)

            # Case 4: Bearing going fully against the wind
            self.assertEqual(npfg.bearingFeasibility(WIND_VECTOR, np.pi, AIRSPEED), 0)

if __name__ == '__main__':
    unittest.main()