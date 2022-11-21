import pytest
from windywings.logger.default_logger import Logger
import numpy as np


def test_euler_to_quaternion():
    input_1 = np.array([0, 0, 0])
    expected_output_1 = np.array([1, 0, 0, 0])
    output_1 = Logger.euler_to_quaternion(*input_1)
    assert np.array_equal(output_1, expected_output_1)

    input_2 = np.array([0, 0, 1])
    expected_output_2 = np.array([0.878, 0, 0, 0.479])
    output_2 = Logger.euler_to_quaternion(*input_2)
    output_2 = np.round(output_2, 3)
    assert np.array_equal(output_2, expected_output_2)

    input_3 = np.array([1, 1, 1])
    expected_output_3 = np.array([0.786, 0.168, 0.571, 0.168])
    output_3 = Logger.euler_to_quaternion(*input_3)
    output_3 = np.round(output_3, 3)
    assert np.array_equal(output_3, expected_output_3)


def test_quaternion_to_euler():
    logger = Logger('results/quaternion_test.csv')

    input_1 = np.array([1, 0, 0, 0])
    expected_output_1 = np.array([0, 0, 0])
    output_1 = Logger.quaternion_to_euler(*input_1)
    assert np.array_equal(output_1, expected_output_1)

    input_2 = np.array([0.878, 0, 0, 0.479])
    expected_output_2 = np.array([0, 0, 1])
    output_2 = Logger.quaternion_to_euler(*input_2)
    output_2 = np.round(output_2, 2)
    assert np.array_equal(output_2, expected_output_2)

    input_3 = np.array([0.786, 0.168, 0.571, 0.168])
    expected_output_3 = np.array([1, 1, 1])
    output_3 = Logger.quaternion_to_euler(*input_3)
    output_3 = np.round(output_3, 2)
    assert np.array_equal(output_3, expected_output_3)
