import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReasonGraph"))

from cot_reasoning import parse_cot_response


class ParseCotResponseTest(unittest.TestCase):
    def test_parses_tagged_steps_and_answer(self):
        parsed = parse_cot_response(
            '<step number="1">Calculate half of 48: 24.</step>\n'
            '<step number="2">Add 48 and 24.</step>\n'
            '<answer>72</answer>',
            "What is the total?",
        )

        self.assertEqual([step.number for step in parsed.steps], [1, 2])
        self.assertEqual(parsed.steps[1].content, "Add 48 and 24.")
        self.assertEqual(parsed.answer, "72")

    def test_parses_plain_numbered_steps_and_final_answer(self):
        parsed = parse_cot_response(
            "1. Understand the question.\n"
            "2. Determine half of 48.\n"
            "Step 1: 48 divided by 2 is 24.\n"
            "3. Add 48 and 24.\n"
            "Final Answer: 72\n\n"
            "Explanation: The result is the sum.",
            "What is the total?",
        )

        self.assertEqual([step.number for step in parsed.steps], [1, 2, 3, 4])
        self.assertEqual(parsed.steps[0].content, "Understand the question.")
        self.assertEqual(parsed.steps[2].content, "48 divided by 2 is 24.")
        self.assertEqual(parsed.steps[3].content, "Add 48 and 24.")
        self.assertEqual(parsed.answer, "72")


if __name__ == "__main__":
    unittest.main()
