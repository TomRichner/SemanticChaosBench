#!/usr/bin/env python3
"""
Runner script for test and demo scripts in the SemanticChaos project.

This script provides a unified interface for running test and demo scripts
with flexible command-line options.

Usage:
    python scripts/run_scripts.py --all              # Run all tests and demos
    python scripts/run_scripts.py --tests            # Run all test scripts
    python scripts/run_scripts.py --demos            # Run all demo scripts
    python scripts/run_scripts.py --scripts test_setup demo_phase1  # Run specific scripts
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple


# Define available scripts
TEST_SCRIPTS = [
    "test_setup.py",
    "test_divergence.py",
    "test_embeddings_and_perturbation.py",
    "test_google_api.py",
    "test_model_interface.py",
    "test_caching.py",
]

DEMO_SCRIPTS = [
    "demo_phase1.py",
    "demo_divergence_with_models.py",
]


class ScriptRunner:
    """Manages execution of test and demo scripts."""
    
    def __init__(self, scripts_dir: Path):
        """
        Initialize the script runner.
        
        Args:
            scripts_dir: Path to the scripts directory
        """
        self.scripts_dir = scripts_dir
        self.tests_dir = scripts_dir / "tests"
        self.demos_dir = scripts_dir / "demos"
        self.results: List[Tuple[str, bool, int]] = []
    
    def run_script(self, script_path: Path, script_name: str) -> Tuple[bool, int]:
        """
        Run a single script and capture its output.
        
        Args:
            script_path: Path to the script file
            script_name: Name of the script for display
            
        Returns:
            Tuple of (success: bool, exit_code: int)
        """
        print("\n" + "=" * 70)
        print(f"Running: {script_name}")
        print("=" * 70)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.scripts_dir.parent,  # Run from project root
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"\n✓ {script_name} completed successfully")
            else:
                print(f"\n✗ {script_name} failed with exit code {result.returncode}")
            
            return success, result.returncode
            
        except Exception as e:
            print(f"\n✗ {script_name} failed with exception: {e}")
            return False, -1
    
    def run_tests(self) -> None:
        """Run all test scripts."""
        print("\n" + "╔" + "=" * 68 + "╗")
        print("║" + " " * 26 + "TEST SCRIPTS" + " " * 30 + "║")
        print("╚" + "=" * 68 + "╝")
        
        for script in TEST_SCRIPTS:
            script_path = self.tests_dir / script
            if script_path.exists():
                success, exit_code = self.run_script(script_path, f"tests/{script}")
                self.results.append((f"tests/{script}", success, exit_code))
            else:
                print(f"\n⚠ Warning: {script} not found in tests directory")
                self.results.append((f"tests/{script}", False, -1))
    
    def run_demos(self) -> None:
        """Run all demo scripts."""
        print("\n" + "╔" + "=" * 68 + "╗")
        print("║" + " " * 26 + "DEMO SCRIPTS" + " " * 30 + "║")
        print("╚" + "=" * 68 + "╝")
        
        for script in DEMO_SCRIPTS:
            script_path = self.demos_dir / script
            if script_path.exists():
                success, exit_code = self.run_script(script_path, f"demos/{script}")
                self.results.append((f"demos/{script}", success, exit_code))
            else:
                print(f"\n⚠ Warning: {script} not found in demos directory")
                self.results.append((f"demos/{script}", False, -1))
    
    def run_specific_scripts(self, script_names: List[str]) -> None:
        """
        Run specific scripts by name.
        
        Args:
            script_names: List of script names (without paths)
        """
        print("\n" + "╔" + "=" * 68 + "╗")
        print("║" + " " * 24 + "SPECIFIC SCRIPTS" + " " * 28 + "║")
        print("╚" + "=" * 68 + "╝")
        
        for script_name in script_names:
            # Normalize script name (add .py if not present)
            if not script_name.endswith('.py'):
                script_name = f"{script_name}.py"
            
            # Check if it's a test script
            if script_name in TEST_SCRIPTS:
                script_path = self.tests_dir / script_name
                display_name = f"tests/{script_name}"
            # Check if it's a demo script
            elif script_name in DEMO_SCRIPTS:
                script_path = self.demos_dir / script_name
                display_name = f"demos/{script_name}"
            else:
                print(f"\n✗ Unknown script: {script_name}")
                print(f"   Available scripts: {', '.join(TEST_SCRIPTS + DEMO_SCRIPTS)}")
                self.results.append((script_name, False, -1))
                continue
            
            if script_path.exists():
                success, exit_code = self.run_script(script_path, display_name)
                self.results.append((display_name, success, exit_code))
            else:
                print(f"\n✗ Script not found: {display_name}")
                self.results.append((display_name, False, -1))
    
    def print_summary(self) -> int:
        """
        Print a summary of all script executions.
        
        Returns:
            Exit code (0 if all passed, 1 if any failed)
        """
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if not self.results:
            print("No scripts were executed.")
            return 0
        
        passed = [r for r in self.results if r[1]]
        failed = [r for r in self.results if not r[1]]
        
        print(f"\nTotal: {len(self.results)} scripts")
        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")
        
        if passed:
            print("\n✓ Passed:")
            for script, _, _ in passed:
                print(f"  - {script}")
        
        if failed:
            print("\n✗ Failed:")
            for script, _, exit_code in failed:
                if exit_code == -1:
                    print(f"  - {script} (not found or exception)")
                else:
                    print(f"  - {script} (exit code: {exit_code})")
        
        print("\n" + "=" * 70)
        
        return 0 if len(failed) == 0 else 1


def main():
    """Main entry point for the script runner."""
    parser = argparse.ArgumentParser(
        description="Run test and demo scripts for SemanticChaos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          Run all tests and demos
  %(prog)s --tests                        Run all test scripts
  %(prog)s --demos                        Run all demo scripts
  %(prog)s --scripts test_setup demo_phase1   Run specific scripts
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--all',
        action='store_true',
        help='Run all test and demo scripts'
    )
    group.add_argument(
        '--tests',
        action='store_true',
        help='Run all test scripts only'
    )
    group.add_argument(
        '--demos',
        action='store_true',
        help='Run all demo scripts only'
    )
    group.add_argument(
        '--scripts',
        nargs='+',
        metavar='SCRIPT',
        help='Run specific scripts by name (e.g., test_setup demo_phase1)'
    )
    
    args = parser.parse_args()
    
    # Determine scripts directory
    scripts_dir = Path(__file__).parent
    
    # Initialize runner
    runner = ScriptRunner(scripts_dir)
    
    # Execute based on arguments
    if args.all:
        runner.run_tests()
        runner.run_demos()
    elif args.tests:
        runner.run_tests()
    elif args.demos:
        runner.run_demos()
    elif args.scripts:
        runner.run_specific_scripts(args.scripts)
    
    # Print summary and exit
    exit_code = runner.print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

