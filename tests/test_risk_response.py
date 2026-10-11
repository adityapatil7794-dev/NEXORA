import unittest
from etl.risk_response_intelligence import build_risk_response


class TestRiskResponse(unittest.TestCase):

    def test_empty_payload(self):
        result = build_risk_response({})

        self.assertEqual(result["status"], "DATA LIMITED")
        self.assertIn("risks", result)
        self.assertIn("recommended_investigations", result)

    def test_result_is_dictionary(self):
        result = build_risk_response({})

        self.assertIsInstance(result, dict)

    def test_result_has_status(self):
        result = build_risk_response({})

        self.assertIn("status", result)


if __name__ == "__main__":
    unittest.main()
    