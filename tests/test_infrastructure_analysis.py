
import unittest
from unittest.mock import patch

from etl.infrastructure_analysis import analyze_infrastructure


class TestInfrastructureAnalysis(unittest.TestCase):

    @patch("etl.infrastructure_analysis._query_overpass")
    def test_all_categories_use_one_query(self, mock_query):
        mock_query.return_value = [
            {
                "type": "node",
                "id": 1,
                "lat": 16.724,
                "lon": 74.544,
                "tags": {"amenity": "hospital", "name": "Test Hospital"},
            },
            {
                "type": "node",
                "id": 2,
                "lat": 16.725,
                "lon": 74.545,
                "tags": {"power": "substation", "name": "Test Substation"},
            },
        ]

        result = analyze_infrastructure(16.723, 74.543, 25)

        mock_query.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["counts"]["hospitals"], 1)
        self.assertEqual(result["counts"]["power_substations"], 1)
        self.assertEqual(result["total_infrastructure_assets"], 2)

    @patch("etl.infrastructure_analysis._query_overpass")
    def test_empty_results_preserve_categories(self, mock_query):
        mock_query.return_value = []

        result = analyze_infrastructure(16.723, 74.543)

        self.assertEqual(len(result["infrastructure"]), 6)
        self.assertEqual(result["total_infrastructure_assets"], 0)

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_infrastructure(100, 74.543)

    def test_invalid_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_infrastructure(16.723, 74.543, 0)


if __name__ == "__main__":
    unittest.main()
