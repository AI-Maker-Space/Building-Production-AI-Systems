#!/usr/bin/env python3
"""
Test runner script for the RAG system.

This script provides different ways to run the test suite with various configurations.
Located in the tests/ directory but runs from the project root.

Examples:
    # From the tests directory:
    python run_tests.py --type fast
    
    # Run only RAG core tests
    python run_tests.py --file tests/test_rag_graph.py
    
    # Run only API tests
    python run_tests.py --file tests/test_app.py
    
    # Run with coverage
    python run_tests.py --type fast --coverage
    
    # Run slow tests
    python run_tests.py --type slow
    
    # From project root:
    python tests/run_tests.py --type fast
"""

import os
import sys
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=True, coverage=False, test_file=None):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ("unit", "all", "fast", "slow")
        verbose: Whether to run in verbose mode
        coverage: Whether to include coverage reporting
        test_file: Specific test file to run (optional)
    """
    # Change to project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    # Configure test selection based on type
    if test_type == "unit":
        cmd.extend(["-m", "not slow"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
        cmd.append("--run-slow")
    elif test_type == "all":
        cmd.append("--run-slow")
    
    # Add test directory or specific file
    if test_file:
        cmd.append(test_file)
    else:
        cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False


def main():
    """Main function to handle command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for the RAG system")
    parser.add_argument(
        "--type", 
        choices=["unit", "all", "fast", "slow"],
        default="fast",
        help="Type of tests to run (default: fast)"
    )
    parser.add_argument(
        "--file",
        help="Specific test file to run (e.g., tests/test_rag_graph.py)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Include coverage reporting"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode (less verbose)"
    )
    
    args = parser.parse_args()
    

    
    # Run the tests
    success = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage,
        test_file=args.file
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
