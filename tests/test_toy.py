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
        Controlled failure drill: Verifies that GitHub Actions CI pipeline
        strictly intercepts failed assertions and flags the build with RED status.
        """
        expected = "EXPECTED_GREEN_STATUS"
        actual = "INTENTIONAL_RED_FAILURE_DRILL"
        self.assertEqual(
            actual,
            expected,
            "Intentional regression drill for Task 6.6: Expecting RED failure on initial run."
        )


if __name__ == "__main__":
    unittest.main()
