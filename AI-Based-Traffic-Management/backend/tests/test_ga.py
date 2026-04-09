import unittest

from optimizer.ga_optimizer import fitness, lane_pressure


class GATests(unittest.TestCase):
    def test_lane_pressure_increases_with_weighted_counts(self):
        low = lane_pressure({"car": 5}, capacity=20)
        high = lane_pressure({"car": 5, "bus": 2}, capacity=20)
        self.assertGreater(high, low)

    def test_fitness_returns_finite_negative_delay(self):
        score = fitness([20, 20, 20, 20], [0.2, 0.3, 0.4, 0.1], 20)
        self.assertTrue(score <= 0)


if __name__ == '__main__':
    unittest.main()
