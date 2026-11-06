"""
Task 6.6: Toy CI Matrix Test Suite
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

This test suite runs in the GitHub Actions matrix workflow across Python 3.10 and 3.11.
Phase 1: Deliberate failure drill to produce a RED build on GitHub Actions.
Phase 2: Fix assertion to achieve a GREEN build.
"""

import unittest


class TestToyMatrix(unittest.TestCase):
    def test_environment_arithmetic(self):
        """Sanity test verifying Python execution environment."""
        self.assertEqual(21 + 21, 42)

    def test_intentional_failure_drill(self):
        """
        Remediated test: Verifies that fixing the assertion yields a GREEN build
        across all matrix environments (Python 3.10 and 3.11).
        """
        expected = "EXPECTED_GREEN_STATUS"
        actual = "EXPECTED_GREEN_STATUS"
        self.assertEqual(
            actual,
            expected,
            "Remediation verification for Task 6.6: Expecting GREEN pass on second run."
        )


if __name__ == "__main__":
    unittest.main()

