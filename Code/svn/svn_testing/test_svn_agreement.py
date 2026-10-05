"""
Subject-Verb Number Agreement Test Script
==========================================
Tests GPT-2 Small's ability to perform correct subject-verb number agreement.

The script evaluates:
1. Baseline performance on correct/incorrect continuations
2. Model accuracy across diverse linguistic patterns
3. Performance consistency across sentence types
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from tqdm import tqdm
from pathlib import Path

# Ensure Code/ is on sys.path so we can import core utilities
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.svn_dataset import get_svn_test_cases, SVNTestCase


class SVNTester:
    """Subject-Verb Number Agreement Tester."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading GPT-2 Small on {device}...")
        self.model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        self.results = []
        
    def create_test_cases(self, limit: int = None) -> List[SVNTestCase]:
        """Return standardized SVN test cases from the shared dataset."""
        return get_svn_test_cases(limit=limit)
    
    def evaluate_exact_match(self, logits: torch.Tensor, correct_token_id: int, 
                            incorrect_token_id: int) -> float:
        """Calculate the logit difference between correct and incorrect tokens."""
        final_logits = logits[0, -1, :]
        return (final_logits[correct_token_id] - final_logits[incorrect_token_id]).item()
    
    def test_single_case(self, test_case: SVNTestCase) -> Dict:
        """Test a single case and return results. Accepts `SVNTestCase` from core.datasets."""
        try:
            # Tokenize correct/incorrect tokens (handle multi-token as before)
            correct_tokens = self.model.to_tokens(test_case.correct_token, prepend_bos=False)
            incorrect_tokens = self.model.to_tokens(test_case.incorrect_token, prepend_bos=False)

            correct_token_id = correct_tokens[0, 0].item()
            incorrect_token_id = incorrect_tokens[0, 0].item()

            # Run baseline on the clean prompt
            baseline_logits, _ = self.model.run_with_cache(test_case.clean_prompt)
            baseline_diff = self.evaluate_exact_match(baseline_logits, correct_token_id, incorrect_token_id)

            # Run corrupted prompt to verify it's wrong
            corrupted_logits, _ = self.model.run_with_cache(test_case.corrupted_prompt)
            corrupted_diff = self.evaluate_exact_match(corrupted_logits, correct_token_id, incorrect_token_id)

            logit_diff_delta = baseline_diff - corrupted_diff
            is_correct = logit_diff_delta > 0

            result = {
                "prompt": test_case.clean_prompt,
                "clean_prompt": test_case.clean_prompt,
                "corrupted_prompt": test_case.corrupted_prompt,
                "subject": test_case.subject,
                "subject_number": test_case.subject_number,
                "correct_verb": test_case.correct_token,
                "incorrect_verb": test_case.incorrect_token,
                "category": test_case.case_type,
                "logit_difference": logit_diff_delta,
                "clean_diff": baseline_diff,
                "corrupted_diff": corrupted_diff,
                "is_correct": is_correct,
                "status": "passed" if is_correct else "failed"
            }
        except Exception as e:
            result = {
                "prompt": getattr(test_case, 'clean_prompt', str(test_case)),
                "clean_prompt": getattr(test_case, 'clean_prompt', None),
                "corrupted_prompt": getattr(test_case, 'corrupted_prompt', None),
                "subject": getattr(test_case, 'subject', ''),
                "subject_number": getattr(test_case, 'subject_number', ''),
                "correct_verb": getattr(test_case, 'correct_token', None),
                "incorrect_verb": getattr(test_case, 'incorrect_token', None),
                "category": getattr(test_case, 'case_type', None),
                "logit_difference": None,
                "is_correct": False,
                "status": f"error: {str(e)}"
            }

        self.results.append(result)
        return result
    
    def run_tests(self, limit: int = None):
        """Run all test cases."""
        print("\n" + "="*80)
        print("SUBJECT-VERB NUMBER AGREEMENT TEST SUITE")
        print("="*80)
        
        test_cases = self.create_test_cases(limit=limit)
        print(f"\nRunning {len(test_cases)} test cases...\n")
        
        with torch.no_grad():
            for test_case in tqdm(test_cases, desc="Running SVN tests"):
                self.test_single_case(test_case)
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*80)
        print("TEST REPORT")
        print("="*80)
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["is_correct"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOverall Results:")
        print(f"  Total Tests:    {total_tests}")
        print(f"  Passed:         {passed_tests}")
        print(f"  Failed:         {failed_tests}")
        print(f"  Pass Rate:      {pass_rate:.1f}%")
        
        # Statistics by category
        print(f"\nResults by Category:")
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        for cat in sorted(categories.keys()):
            stats = categories[cat]
            cat_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {cat:20s}: {stats['passed']:2d}/{stats['total']:2d} ({cat_rate:5.1f}%)")
        
        # Statistics by subject number
        print(f"\nResults by Subject Number:")
        singular_passed = sum(1 for r in self.results if r["subject_number"] == "singular" and r["is_correct"])
        singular_total = sum(1 for r in self.results if r["subject_number"] == "singular")
        plural_passed = sum(1 for r in self.results if r["subject_number"] == "plural" and r["is_correct"])
        plural_total = sum(1 for r in self.results if r["subject_number"] == "plural")
        
        singular_rate = (singular_passed / singular_total * 100) if singular_total > 0 else 0
        plural_rate = (plural_passed / plural_total * 100) if plural_total > 0 else 0
        
        print(f"  Singular:       {singular_passed:2d}/{singular_total:2d} ({singular_rate:5.1f}%)")
        print(f"  Plural:         {plural_passed:2d}/{plural_total:2d} ({plural_rate:5.1f}%)")
        
        # Detailed failure analysis
        failures = [r for r in self.results if not r["is_correct"]]
        if failures:
            print(f"\nFailed Test Cases ({len(failures)}):")
            for result in failures:
                print(f"  ✗ {result['prompt']}")
                print(f"    Subject: {result['subject']} ({result['subject_number']})")
                print(f"    Correct verb: {result['correct_verb']}, "
                      f"Incorrect verb: {result['incorrect_verb']}")
                print(f"    Logit Diff Delta: {result['logit_difference']:.4f} "
                      f"(negative means corruption doesn't flip preference enough)")
                print()
        
        # Logit Delta statistics
        valid_logits = [r["logit_difference"] for r in self.results if r["logit_difference"] is not None]
        if valid_logits:
            print(f"\nLogit Difference Delta Statistics:")
            print(f"  Mean:           {np.mean(valid_logits):.4f}")
            print(f"  Std Dev:        {np.std(valid_logits):.4f}")
            print(f"  Min:            {np.min(valid_logits):.4f}")
            print(f"  Max:            {np.max(valid_logits):.4f}")
        
        print("\n" + "="*80 + "\n")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate,
            "categories": categories,
            "singular_stats": {"passed": singular_passed, "total": singular_total},
            "plural_stats": {"passed": plural_passed, "total": plural_total}
        }
    
    def save_results(self, filename: str = "svn_test_results.json"):
        """Save detailed results to a JSON file."""
        output_path = __import__("pathlib").Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to {output_path}")
    
    def create_visualizations(self):
        """Create a comprehensive set of visualizations."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Overall Pass/Fail Pie Chart
        ax1 = plt.subplot(2, 3, 1)
        total_passed = sum(1 for r in self.results if r["is_correct"])
        total_failed = len(self.results) - total_passed
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax1.pie([total_passed, total_failed], 
                                            labels=['Passed', 'Failed'],
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})
        ax1.set_title('Overall Test Results', fontsize=13, fontweight='bold', pad=20)
        
        # 2. Results by Category
        ax2 = plt.subplot(2, 3, 2)
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["passed"] += 1
        
        cat_names = sorted(categories.keys())
        pass_rates = [categories[cat]["passed"] / categories[cat]["total"] * 100 for cat in cat_names]
        colors_cat = ['#2ecc71' if rate == 100 else '#f39c12' if rate >= 50 else '#e74c3c' for rate in pass_rates]
        
        bars = ax2.barh(cat_names, pass_rates, color=colors_cat, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Pass Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Pass Rate by Category', fontsize=13, fontweight='bold', pad=20)
        ax2.set_xlim(0, 105)
        
        # Add percentage labels
        for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
            ax2.text(rate + 2, i, f'{rate:.0f}%', va='center', fontweight='bold')
        
        # 3. Singular vs Plural Performance
        ax3 = plt.subplot(2, 3, 3)
        singular_passed = sum(1 for r in self.results if r["subject_number"] == "singular" and r["is_correct"])
        singular_total = sum(1 for r in self.results if r["subject_number"] == "singular")
        plural_passed = sum(1 for r in self.results if r["subject_number"] == "plural" and r["is_correct"])
        plural_total = sum(1 for r in self.results if r["subject_number"] == "plural")
        
        singular_rate = singular_passed / singular_total * 100 if singular_total > 0 else 0
        plural_rate = plural_passed / plural_total * 100 if plural_total > 0 else 0
        
        x_pos = np.arange(2)
        rates = [singular_rate, plural_rate]
        colors_number = ['#f39c12' if singular_rate < 100 else '#2ecc71', 
                        '#2ecc71' if plural_rate == 100 else '#f39c12']
        bars = ax3.bar(['Singular', 'Plural'], rates, color=colors_number, edgecolor='black', linewidth=1.5, width=0.6)
        ax3.set_ylabel('Pass Rate (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Performance: Singular vs Plural', fontsize=13, fontweight='bold', pad=20)
        ax3.set_ylim(0, 105)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 4. Logit Difference Distribution
        ax4 = plt.subplot(2, 3, 4)
        valid_logits = [r["logit_difference"] for r in self.results if r["logit_difference"] is not None]
        correct_logits = [r["logit_difference"] for r in self.results if r["is_correct"]]
        incorrect_logits = [r["logit_difference"] for r in self.results if not r["is_correct"]]
        
        ax4.hist(correct_logits, bins=5, alpha=0.7, label='Correct (Passed)', color='#2ecc71', edgecolor='black')
        ax4.hist(incorrect_logits, bins=5, alpha=0.7, label='Incorrect (Failed)', color='#e74c3c', edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax4.set_xlabel('Logit Difference', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Logit Difference Delta Distribution', fontsize=13, fontweight='bold', pad=20)
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(alpha=0.3)
        
        # 5. Test Results Table
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Total Tests', str(len(self.results))],
            ['Passed', f"{total_passed} ({total_passed/len(self.results)*100:.1f}%)"],
            ['Failed', f"{total_failed} ({total_failed/len(self.results)*100:.1f}%)"],
            ['Singular: Passed/Total', f"{singular_passed}/{singular_total}"],
            ['Plural: Passed/Total', f"{plural_passed}/{plural_total}"],
            ['Mean Logit Delta', f"{np.mean(valid_logits):.4f}"],
            ['Std Dev', f"{np.std(valid_logits):.4f}"],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        ax5.set_title('Test Summary Statistics', fontsize=13, fontweight='bold', pad=20)
        
        # 6. Category Performance Stacked Bar
        ax6 = plt.subplot(2, 3, 6)
        cat_names_sorted = sorted(categories.keys())
        passed_counts = [categories[cat]["passed"] for cat in cat_names_sorted]
        failed_counts = [categories[cat]["total"] - categories[cat]["passed"] for cat in cat_names_sorted]
        
        x_pos = np.arange(len(cat_names_sorted))
        p1 = ax6.bar(x_pos, passed_counts, label='Passed', color='#2ecc71', edgecolor='black', linewidth=1)
        p2 = ax6.bar(x_pos, failed_counts, bottom=passed_counts, label='Failed', color='#e74c3c', edgecolor='black', linewidth=1)
        
        ax6.set_ylabel('Test Count', fontsize=11, fontweight='bold')
        ax6.set_title('Test Distribution by Category', fontsize=13, fontweight='bold', pad=20)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(cat_names_sorted, rotation=45, ha='right')
        ax6.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_path = __import__("pathlib").Path(__file__).parent / "svn_test_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()
        
        return output_path
    
    def create_html_report(self):
        """Create a comprehensive HTML report with visualizations."""
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["is_correct"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate visualizations
        viz_path = self.create_visualizations()
        
        # Read visualization as base64
        import base64
        with open(viz_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # Calculate category statistics
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["passed"] += 1
        
        # Calculate subject number statistics
        singular_passed = sum(1 for r in self.results if r["subject_number"] == "singular" and r["is_correct"])
        singular_total = sum(1 for r in self.results if r["subject_number"] == "singular")
        plural_passed = sum(1 for r in self.results if r["subject_number"] == "plural" and r["is_correct"])
        plural_total = sum(1 for r in self.results if r["subject_number"] == "plural")
        
        # Generate failed tests rows
        failed_rows = ""
        for result in self.results:
            if not result["is_correct"]:
                failed_rows += f"""
                <tr style="background-color: #fee;">
                    <td>{result['prompt']}</td>
                    <td>{result['subject']}</td>
                    <td>{result['subject_number']}</td>
                    <td>{result['correct_verb']}</td>
                    <td>{result['incorrect_verb']}</td>
                    <td style="color: red; font-weight: bold;">{result['logit_difference']:.4f}</td>
                </tr>
                """
        
        # Generate category rows
        category_rows = ""
        for cat in sorted(categories.keys()):
            stats = categories[cat]
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            category_rows += f"""
                <tr>
                    <td><strong>{cat}</strong></td>
                    <td>{stats['passed']}</td>
                    <td>{stats['total']}</td>
                    <td>{rate:.1f}%</td>
                </tr>
            """

        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Subject-Verb Number Agreement Test Report</title>
            <style> 
                body {{
                    font-family: Segoe UI, Tahoma, sans-serif;
                    margin: 0;
                    padding: 24px;
                    background: linear-gradient(135deg, #f7f7fb 0%, #eef3ff 100%);
                    color: #1f2937;
                }}
                .card {{
                    max-width: 1280px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
                    overflow: hidden;
                }}
                .header {{
                    padding: 32px 36px;
                    background: linear-gradient(135deg, #1f2937 0%, #334155 100%);
                    color: white;
                }}
                .header h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
                .header p {{ margin: 0; opacity: 0.9; }}
                .content {{
                    padding: 32px 36px 40px 36px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 28px;
                }}
                .stat-card {{
                    padding: 16px;
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    background: #f8fafc;
                }}
                .stat-card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; }}
                .stat-card .value {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
                .stat-card .value.passed {{ color: #2ecc71; }}
                .stat-card .value.failed {{ color: #e74c3c; }}
                .visualization {{
                    text-align: center;
                    margin: 18px 0 30px;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 12px;
                    border: 1px solid #e5e7eb;
                }}
                .section {{
                    margin-top: 30px;
                }}
                .section h2 {{
                    font-size: 22px;
                    color: #1f2937;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 8px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 12px;
                    overflow: hidden;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
                }}
                th {{
                    text-align: left;
                    background: #f8fafc;
                    color: #4b5563;
                    padding: 12px 14px;
                    font-size: 13px;
                    font-weight: 600;
                    border-bottom: 2px solid #e5e7eb;
                }}
                td {{ padding: 12px 14px; border-bottom: 1px solid #e5e7eb; }}
                tr:last-child td {{ border-bottom: none; }}
                tr:nth-child(even) td {{ background: #f8fafc; }}
                .note {{ margin-top: 18px; color: #4b5563; line-height: 1.6; }}
                .footer {{
                    margin-top: 20px;
                    color: #6b7280;
                    font-size: 13px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="header">
                    <h1>Subject-Verb Number Agreement Test Report</h1>
                    <p>Baseline performance evaluation for GPT-2 Small</p>
                </div>
                <div class="content">
                    <div class="summary-grid">
                        <div class="stat-card"><div class="label">Total Tests</div><div class="value">{total_tests}</div></div>
                        <div class="stat-card"><div class="label">Passed</div><div class="value passed">{passed_tests}</div></div>
                        <div class="stat-card"><div class="label">Failed</div><div class="value failed">{failed_tests}</div></div>
                        <div class="stat-card"><div class="label">Pass Rate</div><div class="value">{pass_rate:.1f}%</div></div>
                    </div>

                    <div class="visualization">
                        <img src="data:image/png;base64,{img_data}" alt="SVN Test Visualization" />
                    </div>

                    <div class="section">
                        <h2>Results by Category</h2>
                        <table>
                            <thead><tr><th>Category</th><th>Passed</th><th>Total</th><th>Pass Rate</th></tr></thead>
                            <tbody>{category_rows}</tbody>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Results by Subject Number</h2>
                        <table>
                            <thead><tr><th>Subject Type</th><th>Passed</th><th>Total</th><th>Pass Rate</th></tr></thead>
                            <tbody>
                                <tr>
                                    <td>Singular</td>
                                    <td>{singular_passed}</td>
                                    <td>{singular_total}</td>
                                    <td>{(singular_passed/singular_total*100 if singular_total > 0 else 0):.1f}%</td>
                                </tr>
                                <tr>
                                    <td>Plural</td>
                                    <td>{plural_passed}</td>
                                    <td>{plural_total}</td>
                                    <td>{(plural_passed/plural_total*100 if plural_total > 0 else 0):.1f}%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    {"".join([f'''
                    <div class="section">
                        <h2>Failed Test Cases ({failed_tests})</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Prompt</th>
                                    <th>Subject</th>
                                    <th>Number</th>
                                    <th>Correct Verb</th>
                                    <th>Incorrect Verb</th>
                                    <th>Logit Diff Delta</th>
                                </tr>
                            </thead>
                            <tbody>{failed_rows}</tbody>
                        </table>
                    </div>
                    ''' if failed_tests > 0 else ""])}
                </div>
                <div class="footer">
                    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Save HTML report
        output_path = __import__("pathlib").Path(__file__).parent / "svn_test_report.html"
        with open(output_path, 'w', encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {output_path}")
        
        return output_path


def main():
    """Main entry point for the test suite."""
    import argparse
    parser = argparse.ArgumentParser(description="Run SVN baseline tests.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test cases for a faster run.")
    args = parser.parse_args()

    tester = SVNTester()
    
    # Run all tests
    tester.run_tests(limit=args.limit)
    
    # Generate report
    summary = tester.generate_report()
    
    # Save results
    tester.save_results()
    
    # Create visualizations and HTML report
    print("\nGenerating visualizations...")
    tester.create_html_report()
    
    # Exit with appropriate code
    if summary["failed"] > 0:
        print(f"⚠️  {summary['failed']} test(s) failed")
        return 1
    else:
        print("✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())