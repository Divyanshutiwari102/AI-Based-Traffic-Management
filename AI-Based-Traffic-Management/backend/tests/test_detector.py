import unittest

from detector import weighted_density


class DetectorTests(unittest.TestCase):
    def test_weighted_density_capped(self):
        density = weighted_density({"bus": 100}, capacity=20)
        self.assertEqual(density, 1.0)

    def test_weighted_density_non_negative(self):
        density = weighted_density({}, capacity=20)
        self.assertEqual(density, 0.0)


if __name__ == '__main__':
    unittest.main()
