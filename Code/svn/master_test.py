"""
Master Test Script
==================
Runs the full suite of behavior and ablation method combinations.
"""

from __future__ import annotations
import sys
import argparse
import os
from pathlib import Path

# Fix paths to allow importing from subdirectories
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

# SVN imports
from svn.svn_testing.test_svn_agreement import SVNTester
from svn.mean_ablation.mean_ablation_svn import MeanAblationAnalyzer
from svn.resample_ablation.resample_ablation_svn import ResampleAblationAnalyzer
from svn.zero_ablation.zero_ablation_svn import ZeroAblationAnalyzer
from svn.acdc_svn import ACDCSVNAnalyzer
from svn.causal_scrubbing_svn import CausalScrubber
from svn.aggregate_results import main as aggregate_results_main
from svn.visualize_acdc_svn_dag import main as visualize_acdc_main

def run_svn_baseline(limit: int = None):
    """Runs the SVN behavior baseline metrics."""
    print("="*80)
    print("Running SVN baseline tests...")
    print("="*80)
    tester = SVNTester()
    tester.run_tests(limit=limit)
    tester.generate_report()
    tester.save_results()
    tester.create_html_report()
    print("✓ SVN baseline tests complete!\n")


def run_svn_ablations(limit: int = None):
    """Runs all mapped ablations for the SVN behavior."""
    print("="*80)
    print("Running Zero Ablation for SVN...")
    print("="*80)
    zero_analyzer = ZeroAblationAnalyzer()
    zero_analyzer.run_complete_analysis(limit=limit)
    zero_summary = zero_analyzer.generate_report()
    zero_analyzer.save_results()
    zero_analyzer.create_html_report(zero_summary)
    
    print("\n" + "="*80)
    print("Running Mean Ablation for SVN...")
    print("="*80)
    mean_analyzer = MeanAblationAnalyzer()
    mean_analyzer.run_complete_analysis(limit=limit)
    mean_summary = mean_analyzer.generate_report()
    mean_analyzer.save_results()
    mean_analyzer.create_html_report(mean_summary)

    print("\n" + "="*80)
    print("Running Resample Ablation for SVN...")
    print("="*80)
    resample_analyzer = ResampleAblationAnalyzer()
    resample_analyzer.run_complete_analysis(limit=limit)
    resample_summary = resample_analyzer.generate_report()
    resample_analyzer.save_results()
    resample_analyzer.create_html_report(resample_summary)
    
    print("\n✓ All ablations complete!\n")


def run_circuit_discovery(limit: int = None, threshold: float = 0.3):
    """Runs ACDC to discover the minimal SVN circuit."""
    print("\n" + "="*80)
    print("Running ACDC Circuit Discovery for SVN...")
    print("="*80)
    acdc_analyzer = ACDCSVNAnalyzer()
    acdc_analyzer.run_acdc_discovery(threshold=threshold, limit=limit)
    acdc_summary = acdc_analyzer.generate_report()
    if acdc_summary:
        acdc_analyzer.save_results()
        acdc_analyzer.create_html_report(acdc_summary)
        print("\n✓ ACDC circuit discovery complete!\n")

def run_circuit_visualization():
    """Visualizes the discovered ACDC circuit as a DAG."""
    print("\n" + "="*80)
    print("Visualizing ACDC Circuit for SVN...")
    print("="*80)
    visualize_acdc_main()
    print("\n✓ ACDC circuit visualization complete!\n")


def run_causal_scrubbing(limit: int = None):
    """Runs Causal Scrubbing to validate the discovered circuit."""
    print("\n" + "="*80)
    print("Running Causal Scrubbing for SVN...")
    print("="*80)
    try:
        scrubber = CausalScrubber()
        scrubber.run_scrubbing(limit=limit)
        print("\n✓ Causal scrubbing validation complete!\n")
    except FileNotFoundError as e:
        print(f"\n[!] Error during Causal Scrubbing: {e}")
        print("Make sure ACDC has completed successfully first.")


def run_aggregation():
    """Aggregates all results into a summary table and visualization."""
    print("\n" + "="*80)
    print("Aggregating Results...")
    print("="*80)
    aggregate_results_main()
    print("\n✓ Results aggregation complete!\n")
    

def main():
    parser = argparse.ArgumentParser(description="Run the master SVN analysis suite.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test cases for a faster run.")
    parser.add_argument("--threshold", type=float, default=0.3, help="ACDC pruning threshold.")
    args = parser.parse_args()

    limit_str = f"(Limit: {args.limit} cases)" if args.limit else "(Full Dataset)"
    print(f"Starting Master SVN Analysis Suite... {limit_str}")
    
    # 1. Run Baseline Tests
    run_svn_baseline(limit=args.limit)
    
    # 2. Run Ablations
    run_svn_ablations(limit=args.limit)
    
    # 3. Run Circuit Discovery (ACDC)
    run_circuit_discovery(limit=args.limit, threshold=args.threshold)
    
    # 4. Visualize Discovered Circuit
    run_circuit_visualization()
    
    # 5. Run Circuit Validation (Causal Scrubbing)
    run_causal_scrubbing(limit=args.limit)
    
    # 6. Aggregate Final Results
    run_aggregation()
    
    print("="*80)
    print("Master test suite complete! All analyses, circuit discovery, validations, and visual aggregations have been processed successfully.")
    print("="*80)

    
if __name__ == "__main__":
    main()
