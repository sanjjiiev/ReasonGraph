import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReasonGraph"))

import app as application


class StubAPI:
    def generate_response(self, *args, **kwargs):
        return "Unstructured output that has no parseable reasoning."


class BenchmarkEndpointTest(unittest.TestCase):
    def test_unparseable_output_is_not_awarded_a_graph_score(self):
        client = application.app.test_client()
        with patch.object(application, "create_api", return_value=StubAPI()), \
                patch.object(application, "_load_benchmark_results", return_value=[]), \
                patch.object(application, "_save_benchmark_results"), \
                patch.object(
                    application.graph_evaluator, "evaluate_graph"
                ) as evaluate_graph:
            response = client.post(
                "/api/benchmark",
                json={
                    "question": "What is 48 plus half of 48?",
                    "reasoning_method": "cot",
                    "models": [
                        {
                            "model": "test-model",
                            "provider": "huggingface",
                            "api_key": "",
                        }
                    ],
                },
            )

        result = response.get_json()["results"][0]
        self.assertIsNone(result["rqi"])
        self.assertEqual(
            result["error"],
            "Graph extraction produced no reasoning or answer nodes",
        )
        evaluate_graph.assert_not_called()


if __name__ == "__main__":
    unittest.main()
