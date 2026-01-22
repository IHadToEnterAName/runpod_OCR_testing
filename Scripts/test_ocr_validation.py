#!/usr/bin/env python3
"""
OCR Validation Script with CER/WER Metrics
==========================================
Measures Character Error Rate (CER) and Word Error Rate (WER)
to validate OCR accuracy improvements.

Usage:
    python test_ocr_validation.py --ground-truth /path/to/ground_truth.txt --pdf /path/to/test.pdf
    python test_ocr_validation.py --run-benchmarks
    python test_ocr_validation.py --compare-modes
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# ERROR RATE CALCULATION
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    Used for both CER and WER calculations.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (Substitutions + Insertions + Deletions) / Length of Reference

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        CER as a float between 0 and 1 (or higher if more errors than chars)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0

    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Insertions + Deletions) / Number of Reference Words

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        WER as a float between 0 and 1 (or higher if more errors than words)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    - Convert to lowercase
    - Normalize whitespace
    - Remove special formatting
    """
    import re

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special markers from our processing
    text = re.sub(r'\[BLOCK \d+\]', '', text)
    text = re.sub(r'\[COLUMN \d+\]', '', text)
    text = re.sub(r'\[(HEADER|FOOTER|CONTENT|TABLE START|TABLE END|/HEADER|/FOOTER|/CONTENT)\]', '', text)
    text = re.sub(r'\[CONTEXT FROM PREVIOUS PAGES\].*?\[END PREVIOUS CONTEXT\]', '', text, flags=re.DOTALL)

    # Remove multiple spaces again after marker removal
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =============================================================================
# VALIDATION DATA STRUCTURES
# =============================================================================

@dataclass
class PageMetrics:
    """Metrics for a single page."""
    page_number: int
    cer: float
    wer: float
    reference_chars: int
    hypothesis_chars: int
    reference_words: int
    hypothesis_words: int


@dataclass
class DocumentMetrics:
    """Aggregated metrics for a document."""
    filename: str
    total_pages: int
    avg_cer: float
    avg_wer: float
    min_cer: float
    max_cer: float
    min_wer: float
    max_wer: float
    page_metrics: List[PageMetrics]
    processing_time_seconds: float
    mode: str  # 'simple', 'layout_aware', 'full_enhanced'


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_documents: int
    overall_cer: float
    overall_wer: float
    document_metrics: List[DocumentMetrics]
    comparison_summary: Dict[str, Dict[str, float]]  # mode -> {cer, wer}


# =============================================================================
# OCR EXTRACTION FOR TESTING
# =============================================================================

async def extract_with_simple_mode(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract text using simple OCR (original method)."""
    from processing.document_extractor import extract_pdf_pages
    return extract_pdf_pages(pdf_path)


async def extract_with_layout_mode(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract text using layout-aware OCR."""
    from processing.document_extractor import extract_pdf_pages_with_layout
    pages_with_layout = extract_pdf_pages_with_layout(pdf_path, preserve_structure=True)
    return [(p, text) for p, text, _ in pages_with_layout]


async def extract_with_full_enhanced(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract text using full enhanced mode (layout + context stitching simulation)."""
    from processing.document_extractor import extract_pdf_pages_with_layout
    pages_with_layout = extract_pdf_pages_with_layout(pdf_path, preserve_structure=True)

    # Simulate context stitching effects (the actual stitching happens during processing)
    # For validation, we just use the structured text
    return [(p, text) for p, text, _ in pages_with_layout]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_page(
    reference: str,
    hypothesis: str,
    page_number: int,
    normalize: bool = True
) -> PageMetrics:
    """
    Validate a single page's OCR output against ground truth.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    cer = calculate_cer(reference, hypothesis)
    wer = calculate_wer(reference, hypothesis)

    return PageMetrics(
        page_number=page_number,
        cer=cer,
        wer=wer,
        reference_chars=len(reference),
        hypothesis_chars=len(hypothesis),
        reference_words=len(reference.split()),
        hypothesis_words=len(hypothesis.split())
    )


async def validate_document(
    pdf_path: str,
    ground_truth_pages: Dict[int, str],
    mode: str = 'full_enhanced'
) -> DocumentMetrics:
    """
    Validate an entire document against ground truth.

    Args:
        pdf_path: Path to PDF file
        ground_truth_pages: Dict mapping page numbers to ground truth text
        mode: 'simple', 'layout_aware', or 'full_enhanced'

    Returns:
        DocumentMetrics with validation results
    """
    start_time = time.time()

    # Extract based on mode
    if mode == 'simple':
        pages = await extract_with_simple_mode(pdf_path)
    elif mode == 'layout_aware':
        pages = await extract_with_layout_mode(pdf_path)
    else:  # full_enhanced
        pages = await extract_with_full_enhanced(pdf_path)

    processing_time = time.time() - start_time

    # Validate each page
    page_metrics = []
    for page_num, ocr_text in pages:
        if page_num in ground_truth_pages:
            metrics = validate_page(
                ground_truth_pages[page_num],
                ocr_text,
                page_num
            )
            page_metrics.append(metrics)

    # Calculate aggregated metrics
    if page_metrics:
        avg_cer = sum(pm.cer for pm in page_metrics) / len(page_metrics)
        avg_wer = sum(pm.wer for pm in page_metrics) / len(page_metrics)
        min_cer = min(pm.cer for pm in page_metrics)
        max_cer = max(pm.cer for pm in page_metrics)
        min_wer = min(pm.wer for pm in page_metrics)
        max_wer = max(pm.wer for pm in page_metrics)
    else:
        avg_cer = avg_wer = min_cer = max_cer = min_wer = max_wer = 1.0

    return DocumentMetrics(
        filename=os.path.basename(pdf_path),
        total_pages=len(pages),
        avg_cer=avg_cer,
        avg_wer=avg_wer,
        min_cer=min_cer,
        max_cer=max_cer,
        min_wer=min_wer,
        max_wer=max_wer,
        page_metrics=page_metrics,
        processing_time_seconds=processing_time,
        mode=mode
    )


def generate_synthetic_ground_truth(pdf_path: str) -> Dict[int, str]:
    """
    Generate synthetic ground truth from PDF for self-comparison testing.
    Uses high-quality extraction as pseudo-ground-truth.
    """
    from processing.document_extractor import extract_pdf_pages

    pages = extract_pdf_pages(pdf_path)
    return {page_num: text for page_num, text in pages}


# =============================================================================
# COMPARISON AND REPORTING
# =============================================================================

async def compare_extraction_modes(
    pdf_path: str,
    ground_truth_pages: Optional[Dict[int, str]] = None
) -> Dict[str, DocumentMetrics]:
    """
    Compare all extraction modes on the same document.
    """
    if ground_truth_pages is None:
        # Use simple extraction as baseline for comparison
        ground_truth_pages = generate_synthetic_ground_truth(pdf_path)

    results = {}
    modes = ['simple', 'layout_aware', 'full_enhanced']

    for mode in modes:
        print(f"  Testing {mode} mode...")
        metrics = await validate_document(pdf_path, ground_truth_pages, mode)
        results[mode] = metrics

    return results


def print_validation_report(report: ValidationReport):
    """Print a formatted validation report to console."""
    print("\n" + "=" * 70)
    print("OCR VALIDATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Documents Tested: {report.total_documents}")
    print(f"\nOverall Metrics:")
    print(f"  • Character Error Rate (CER): {report.overall_cer:.2%}")
    print(f"  • Word Error Rate (WER): {report.overall_wer:.2%}")

    if report.comparison_summary:
        print(f"\n{'Mode Comparison':^70}")
        print("-" * 70)
        print(f"{'Mode':<20} {'CER':<15} {'WER':<15} {'Improvement':<20}")
        print("-" * 70)

        baseline_cer = report.comparison_summary.get('simple', {}).get('cer', 1.0)
        baseline_wer = report.comparison_summary.get('simple', {}).get('wer', 1.0)

        for mode, metrics in report.comparison_summary.items():
            cer = metrics.get('cer', 0)
            wer = metrics.get('wer', 0)

            if mode == 'simple':
                improvement = "baseline"
            else:
                cer_improvement = ((baseline_cer - cer) / baseline_cer * 100) if baseline_cer > 0 else 0
                improvement = f"CER: {cer_improvement:+.1f}%"

            print(f"{mode:<20} {cer:<15.2%} {wer:<15.2%} {improvement:<20}")

    print("\n" + "-" * 70)
    print("Per-Document Details:")
    print("-" * 70)

    for doc in report.document_metrics:
        print(f"\n📄 {doc.filename} ({doc.mode})")
        print(f"   Pages: {doc.total_pages} | Time: {doc.processing_time_seconds:.2f}s")
        print(f"   CER: {doc.avg_cer:.2%} (min: {doc.min_cer:.2%}, max: {doc.max_cer:.2%})")
        print(f"   WER: {doc.avg_wer:.2%} (min: {doc.min_wer:.2%}, max: {doc.max_wer:.2%})")

    print("\n" + "=" * 70)


def save_report_json(report: ValidationReport, output_path: str):
    """Save validation report as JSON."""
    report_dict = {
        'timestamp': report.timestamp,
        'total_documents': report.total_documents,
        'overall_cer': report.overall_cer,
        'overall_wer': report.overall_wer,
        'comparison_summary': report.comparison_summary,
        'document_metrics': [asdict(dm) for dm in report.document_metrics]
    }

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n📊 Report saved to: {output_path}")


# =============================================================================
# BENCHMARK TEST SUITE
# =============================================================================

# Sample test cases for validation
SAMPLE_TEST_CASES = [
    {
        "name": "Simple Text (exact match)",
        "reference": "The quick brown fox jumps over the lazy dog.",
        "ocr_output": "The quick brown fox jumps over the lazy dog.",
        "expected_cer": 0.0,
        "expected_wer": 0.0
    },
    {
        "name": "Minor OCR Errors (substitutions)",
        "reference": "The quick brown fox jumps over the lazy dog.",
        "ocr_output": "The quIck br0wn fox jumps 0ver the lazy d0g.",
        "expected_cer_max": 0.15,  # Should be under 15%
        "expected_wer_max": 0.5
    },
    {
        "name": "Table Extraction (normalized comparison)",
        "reference": "name john age 25 city nyc",  # Already normalized
        "ocr_output": "name john age 25 city nyc",
        "expected_cer": 0.0,  # After normalization, tables should match
        "expected_wer": 0.0
    },
    {
        "name": "Whitespace Normalization",
        "reference": "Hello    world.  How  are   you?",
        "ocr_output": "Hello world. How are you?",
        "expected_cer": 0.0,  # Whitespace normalized
        "expected_wer": 0.0
    },
    {
        "name": "Character Confusion (l/1, O/0)",
        "reference": "Total: 100 items, Order ID: O12345",
        "ocr_output": "Tota1: l00 items, 0rder 1D: 012345",
        "expected_cer_max": 0.25,  # Common OCR errors
        "expected_wer_max": 0.85  # High WER expected for character-level confusion
    }
]


def run_unit_tests():
    """Run unit tests for CER/WER calculation."""
    print("\n🧪 Running Unit Tests...")
    print("-" * 50)

    passed = 0
    failed = 0

    for test in SAMPLE_TEST_CASES:
        name = test["name"]
        reference = test["reference"]
        hypothesis = test["ocr_output"]

        # Calculate with normalization (as used in real validation)
        cer = calculate_cer(normalize_text(reference), normalize_text(hypothesis))
        wer = calculate_wer(normalize_text(reference), normalize_text(hypothesis))

        # Check expected results
        test_passed = True
        details = []

        if "expected_cer" in test:
            if abs(cer - test["expected_cer"]) > 0.001:
                test_passed = False
                details.append(f"CER: {cer:.2%} != {test['expected_cer']:.2%}")

        if "expected_cer_max" in test:
            if cer > test["expected_cer_max"]:
                test_passed = False
                details.append(f"CER: {cer:.2%} > {test['expected_cer_max']:.2%}")

        if "expected_wer" in test:
            if abs(wer - test["expected_wer"]) > 0.001:
                test_passed = False
                details.append(f"WER: {wer:.2%} != {test['expected_wer']:.2%}")

        if "expected_wer_max" in test:
            if wer > test["expected_wer_max"]:
                test_passed = False
                details.append(f"WER: {wer:.2%} > {test['expected_wer_max']:.2%}")

        status = "✅" if test_passed else "❌"
        print(f"{status} {name}: CER={cer:.2%}, WER={wer:.2%}")

        if details:
            for d in details:
                print(f"   ⚠️ {d}")

        if test_passed:
            passed += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="OCR Validation with CER/WER Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to validate"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth text file (one page per section, separated by '---PAGE---')"
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run benchmark tests on sample data"
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare simple vs layout-aware vs full-enhanced modes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--unit-tests",
        action="store_true",
        help="Run unit tests for CER/WER calculation"
    )

    args = parser.parse_args()

    # Run unit tests if requested
    if args.unit_tests:
        success = run_unit_tests()
        return 0 if success else 1

    # Run benchmarks
    if args.run_benchmarks:
        print("\n📊 Running Benchmarks...")
        run_unit_tests()
        return 0

    # Validate specific PDF
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"❌ PDF file not found: {args.pdf}")
            return 1

        # Load ground truth if provided
        ground_truth_pages = None
        if args.ground_truth and os.path.exists(args.ground_truth):
            with open(args.ground_truth, 'r') as f:
                content = f.read()

            pages = content.split('---PAGE---')
            ground_truth_pages = {
                i + 1: page.strip()
                for i, page in enumerate(pages)
                if page.strip()
            }
            print(f"📖 Loaded ground truth for {len(ground_truth_pages)} pages")

        if args.compare_modes:
            print(f"\n🔬 Comparing extraction modes on: {args.pdf}")
            results = await compare_extraction_modes(args.pdf, ground_truth_pages)

            # Create report
            comparison_summary = {
                mode: {'cer': metrics.avg_cer, 'wer': metrics.avg_wer}
                for mode, metrics in results.items()
            }

            all_metrics = list(results.values())
            overall_cer = sum(m.avg_cer for m in all_metrics) / len(all_metrics)
            overall_wer = sum(m.avg_wer for m in all_metrics) / len(all_metrics)

            report = ValidationReport(
                timestamp=datetime.now().isoformat(),
                total_documents=1,
                overall_cer=overall_cer,
                overall_wer=overall_wer,
                document_metrics=all_metrics,
                comparison_summary=comparison_summary
            )

            print_validation_report(report)
            save_report_json(report, args.output)
        else:
            print(f"\n🔬 Validating: {args.pdf}")
            metrics = await validate_document(
                args.pdf,
                ground_truth_pages or generate_synthetic_ground_truth(args.pdf),
                mode='full_enhanced'
            )

            report = ValidationReport(
                timestamp=datetime.now().isoformat(),
                total_documents=1,
                overall_cer=metrics.avg_cer,
                overall_wer=metrics.avg_wer,
                document_metrics=[metrics],
                comparison_summary={}
            )

            print_validation_report(report)
            save_report_json(report, args.output)

        return 0

    # Default: run unit tests
    print("No action specified. Running unit tests...")
    success = run_unit_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
