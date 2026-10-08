import unittest
from math_checker import check_math

class TestMathChecker(unittest.TestCase):
    def test_correct_math(self):
        text = "The total is 10 + 5 = 15."
        errors = check_math(text)
        self.assertEqual(len(errors), 0)
        
    def test_incorrect_math(self):
        text = "Since she sold half as many, 48 / 2 = 25 clips."
        errors = check_math(text)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['type'], 'arithmetic')
        self.assertTrue('48 / 2 = 24' in errors[0]['reason'])
        self.assertTrue('not 25' in errors[0]['reason'])
        
    def test_multiple_equations(self):
        text = "First, 10 * 2 = 20. Then 20 - 5 = 14."
        errors = check_math(text)
        self.assertEqual(len(errors), 1)
        self.assertTrue('20 - 5 = 15' in errors[0]['reason'])
        
    def test_decimals(self):
        text = "The cost is 2.5 * 4 = 11.0."
        errors = check_math(text)
        self.assertEqual(len(errors), 1)
        self.assertTrue('2.5 * 4 = 10' in errors[0]['reason'])
        
if __name__ == '__main__':
    unittest.main()
