"""
Test runner script for GNN DDI model tests
"""

import sys
import pytest
from pathlib import Path

def run_tests():
    """Run all GNN model tests"""
    test_dir = Path(__file__).parent

    print("="*70)
    print("Running GNN DDI Model Tests")
    print("="*70)
    print()

    # Run tests with verbose output
    exit_code = pytest.main([
        str(test_dir / 'test_gnn.py'),
        '-v',
        '--tb=short',
        '--color=yes',
        '-s'
    ])

    if exit_code == 0:
        print()
        print("="*70)
        print("All tests passed!")
        print("="*70)
    else:
        print()
        print("="*70)
        print("Some tests failed. Please review the output above.")
        print("="*70)

    return exit_code

if __name__ == '__main__':
    sys.exit(run_tests())
