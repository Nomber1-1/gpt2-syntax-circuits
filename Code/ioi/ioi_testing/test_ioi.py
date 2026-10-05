"""
Indirect Object Identification (IOI) Test Script
================================================
Tests GPT-2 Small's ability to perform correct entity tracking and IOI.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Dict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from core.ioi_dataset import get_ioi_test_cases, IOITestCase

class IOITester:
    """Indirect Object Identification Tester."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading GPT-2 Small on {device}...")
        self.model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        self.results = []
        
    def create_test_cases(self) -> List[IOITestCase]:
        return get_ioi_test_cases()
    
    def evaluate_exact_match(self, logits: torch.Tensor, correct_token_id: int, 
                            incorrect_token_id: int) -> float:
        final_logits = logits[0, -1, :]
        return (final_logits[correct_token_id] - final_logits[incorrect_token_id]).item()
    
    def test_single_case(self, test_case: IOITestCase) -> Dict:
        try:
            correct_tokens = self.model.to_tokens(test_case.correct_token, prepend_bos=False)
            incorrect_tokens = self.model.to_tokens(test_case.incorrect_token, prepend_bos=False)

            correct_token_id = correct_tokens[0, 0].item()
            incorrect_token_id = incorrect_tokens[0, 0].item()

            baseline_logits, _ = self.model.run_with_cache(test_case.clean_prompt)
            baseline_diff = self.evaluate_exact_match(baseline_logits, correct_token_id, incorrect_token_id)

            corrupted_logits, _ = self.model.run_with_cache(test_case.corrupted_prompt)
            corrupted_diff = self.evaluate_exact_match(corrupted_logits, correct_token_id, incorrect_token_id)

            logit_diff_delta = baseline_diff - corrupted_diff
            is_correct = logit_diff_delta > 0

            result = {
                "prompt": test_case.clean_prompt,
                "clean_prompt": test_case.clean_prompt,
                "corrupted_prompt": test_case.corrupted_prompt,
                "subject_io": test_case.subject_io,
                "subject_s": test_case.subject_s,
                "correct_token": test_case.correct_token,
                "incorrect_token": test_case.incorrect_token,
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
                "subject_io": getattr(test_case, 'subject_io', ''),
                "subject_s": getattr(test_case, 'subject_s', ''),
                "correct_token": getattr(test_case, 'correct_token', None),
                "incorrect_token": getattr(test_case, 'incorrect_token', None),
                "category": getattr(test_case, 'case_type', None),
                "logit_difference": None,
                "is_correct": False,
                "status": f"error: {str(e)}"
            }

        self.results.append(result)
        return result
    
    def run_tests(self):
        print("\n" + "="*80)
        print("INDIRECT OBJECT IDENTIFICATION (IOI) TEST SUITE")
        print("="*80)
        
        test_cases = self.create_test_cases()
        print(f"\nRunning {len(test_cases)} test cases...\n")
        
        for idx, test_case in enumerate(test_cases, 1):
            result = self.test_single_case(test_case)
            status_symbol = "✓" if result["is_correct"] else "✗"
            logit_str = f"{result['logit_difference']:7.4f}" if result['logit_difference'] is not None else "  ERROR"
            print(f"[{idx:2d}] {status_symbol} [{result['category']:5s}] {result['prompt']:45s} "
                  f"| Logit Diff Delta: {logit_str}")
    
    def generate_report(self):
        print("\n" + "="*80)
        print("TEST REPORT")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["is_correct"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOverall Results:")
        print(f"  Total Tests:    {total_tests}")
        print(f"  Passed:         {passed_tests}")
        print(f"  Failed:         {failed_tests}")
        print(f"  Pass Rate:      {pass_rate:.1f}%")
        
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
        
        failures = [r for r in self.results if not r["is_correct"]]
        if failures:
            print(f"\nFailed Test Cases ({len(failures)}):")
            for result in failures:
                print(f"  ✗ {result['prompt']}")
                print(f"    IO: {result['subject_io']}, Subject: {result['subject_s']}")
                print(f"    Correct token: {result['correct_token']}, "
                      f"Incorrect token: {result['incorrect_token']}")
                print(f"    Logit Diff Delta: {result['logit_difference']:.4f}")
                print()
        
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
        }
    
    def save_results(self, filename: str = "ioi_test_results.json"):
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to {output_path}")
    
    def create_visualizations(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(2, 2, 1)
        total_passed = sum(1 for r in self.results if r["is_correct"])
        total_failed = len(self.results) - total_passed
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie([total_passed, total_failed], labels=['Passed', 'Failed'],
                autopct='%1.1f%%', colors=colors, startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
        ax1.set_title('Overall Test Results', fontsize=13, fontweight='bold', pad=20)
        
        ax2 = plt.subplot(2, 2, 2)
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
        ax2.set_title('Pass Rate by Structure Category', fontsize=13, fontweight='bold', pad=20)
        ax2.set_xlim(0, 105)
        for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
            ax2.text(rate + 2, i, f'{rate:.0f}%', va='center', fontweight='bold')
        
        ax4 = plt.subplot(2, 2, 3)
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
        
        ax5 = plt.subplot(2, 2, 4)
        ax5.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Total Tests', str(len(self.results))],
            ['Passed', f"{total_passed} ({total_passed/len(self.results)*100:.1f}%)"],
            ['Failed', f"{total_failed} ({total_failed/len(self.results)*100:.1f}%)"],
            ['Mean Logit Delta', f"{np.mean(valid_logits):.4f}"],
            ['Std Dev', f"{np.std(valid_logits):.4f}"],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        ax5.set_title('Test Summary Statistics', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = Path(__file__).parent / "ioi_test_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()
        return output_path
    
    def create_html_report(self):
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["is_correct"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        viz_path = self.create_visualizations()
        
        import base64
        with open(viz_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["passed"] += 1
        
        failed_rows = ""
        for result in self.results:
            if not result["is_correct"]:
                failed_rows += f"""
                <tr style="background-color: #fee;">
                    <td>{result['prompt']}</td>
                    <td>{result['subject_s']}</td>
                    <td>{result['subject_io']}</td>
                    <td>{result['correct_token']}</td>
                    <td>{result['incorrect_token']}</td>
                    <td style="color: red; font-weight: bold;">{result['logit_difference']:.4f}</td>
                </tr>
                """
        
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
            <title>IOI Test Report</title>
            <style> 
                body {{ font-family: Segoe UI, Tahoma, sans-serif; margin: 0; padding: 24px; background: #f7f7fb; }}
                .card {{ max-width: 1280px; margin: 0 auto; background: white; border-radius: 16px; padding: 30px; box-shadow: 0 18px 50px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #1f2937; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
                th, td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
                th {{ background: #f8fafc; }}
                img {{ max-width: 100%; border-radius: 12px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Indirect Object Identification Test Report</h1>
                <p>Total Tests: {total_tests} | Pass Rate: {pass_rate:.1f}%</p>
                
                <div style="text-align:center; margin: 20px 0;">
                    <img src="data:image/png;base64,{img_data}" alt="Visualization" />
                </div>

                <h2>Results by Category (ABA vs BBA)</h2>
                <table>
                    <thead><tr><th>Category</th><th>Passed</th><th>Total</th><th>Pass Rate</th></tr></thead>
                    <tbody>{category_rows}</tbody>
                </table>

                {"<h2>Failed Cases</h2><table><thead><tr><th>Prompt</th><th>Subject (S)</th><th>Target (IO)</th><th>Correct Token</th><th>Incorrect Token</th><th>Delta</th></tr></thead><tbody>" + failed_rows + "</tbody></table>" if failed_tests > 0 else ""}
            </div>
        </body>
        </html>
        '''
        
        output_path = Path(__file__).parent / "ioi_test_report.html"
        with open(output_path, 'w', encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {output_path}")
        
        return output_path

def main():
    tester = IOITester()
    tester.run_tests()
    summary = tester.generate_report()
    tester.save_results()
    
    print("\nGenerating visualizations...")
    tester.create_html_report()
    
    if summary["failed"] > 0:
        print(f"⚠️  {summary['failed']} test(s) failed")
        return 1
    else:
        print("✓ All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())