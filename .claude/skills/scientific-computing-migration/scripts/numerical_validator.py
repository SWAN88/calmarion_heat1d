#!/usr/bin/env python3
"""
Numerical Validator for Scientific Computing Migration

Compares Python/JAX implementation against reference data (Fortran/MATLAB/CalculiX).

Usage:
    python numerical_validator.py --jax result.npy --reference ref.npy --rtol 1e-10
    python numerical_validator.py --jax-dir jax_outputs/ --ref-dir fortran_outputs/
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import numpy as np

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@dataclass
class ValidationResult:
    """Result of validating a single comparison."""
    name: str
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    rms_error: float
    rtol: float
    atol: float
    shape: tuple
    dtype: str
    notes: Optional[str] = None


@dataclass 
class ValidationReport:
    """Complete validation report."""
    results: List[ValidationResult]
    summary: Dict
    
    def to_dict(self):
        return {
            'results': [asdict(r) for r in self.results],
            'summary': self.summary
        }


def load_array(filepath: Path) -> np.ndarray:
    """Load array from various formats."""
    suffix = filepath.suffix.lower()
    
    if suffix == '.npy':
        return np.load(filepath)
    elif suffix == '.npz':
        data = np.load(filepath)
        # Return first array in npz
        return data[list(data.keys())[0]]
    elif suffix == '.txt' or suffix == '.dat':
        return np.loadtxt(filepath)
    elif suffix == '.csv':
        return np.loadtxt(filepath, delimiter=',')
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def validate_arrays(
    result: np.ndarray,
    reference: np.ndarray,
    name: str,
    rtol: float = 1e-10,
    atol: float = 1e-12
) -> ValidationResult:
    """
    Validate result against reference within tolerance.
    
    Args:
        result: Array from Python/JAX implementation
        reference: Reference array from Fortran/MATLAB
        name: Name for this comparison
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        ValidationResult with detailed metrics
    """
    # Ensure same shape
    if result.shape != reference.shape:
        return ValidationResult(
            name=name,
            passed=False,
            max_absolute_error=float('inf'),
            max_relative_error=float('inf'),
            mean_absolute_error=float('inf'),
            rms_error=float('inf'),
            rtol=rtol,
            atol=atol,
            shape=result.shape,
            dtype=str(result.dtype),
            notes=f"Shape mismatch: {result.shape} vs {reference.shape}"
        )
    
    # Compute errors
    abs_error = np.abs(result - reference)
    
    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(reference), atol)
    rel_error = abs_error / denom
    
    max_abs = float(np.max(abs_error))
    max_rel = float(np.max(rel_error))
    mean_abs = float(np.mean(abs_error))
    rms = float(np.sqrt(np.mean(abs_error**2)))
    
    # Check if within tolerance
    passed = np.allclose(result, reference, rtol=rtol, atol=atol)
    
    # Find location of max error
    max_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    notes = f"Max error at index {max_idx}: result={result[max_idx]:.15e}, ref={reference[max_idx]:.15e}"
    
    return ValidationResult(
        name=name,
        passed=passed,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        mean_absolute_error=mean_abs,
        rms_error=rms,
        rtol=rtol,
        atol=atol,
        shape=result.shape,
        dtype=str(result.dtype),
        notes=notes
    )


def validate_files(
    result_path: Path,
    reference_path: Path,
    rtol: float = 1e-10,
    atol: float = 1e-12
) -> ValidationResult:
    """Validate a single pair of files."""
    result = load_array(result_path)
    reference = load_array(reference_path)
    name = result_path.stem
    return validate_arrays(result, reference, name, rtol, atol)


def validate_directories(
    result_dir: Path,
    reference_dir: Path,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    pattern: str = "*.npy"
) -> List[ValidationResult]:
    """Validate all matching files in directories."""
    results = []
    
    result_files = sorted(result_dir.glob(pattern))
    
    for result_path in result_files:
        # Find matching reference file
        ref_path = reference_dir / result_path.name
        
        if not ref_path.exists():
            results.append(ValidationResult(
                name=result_path.stem,
                passed=False,
                max_absolute_error=float('inf'),
                max_relative_error=float('inf'),
                mean_absolute_error=float('inf'),
                rms_error=float('inf'),
                rtol=rtol,
                atol=atol,
                shape=(0,),
                dtype='unknown',
                notes=f"Reference file not found: {ref_path}"
            ))
            continue
        
        try:
            result = validate_files(result_path, ref_path, rtol, atol)
            results.append(result)
        except Exception as e:
            results.append(ValidationResult(
                name=result_path.stem,
                passed=False,
                max_absolute_error=float('inf'),
                max_relative_error=float('inf'),
                mean_absolute_error=float('inf'),
                rms_error=float('inf'),
                rtol=rtol,
                atol=atol,
                shape=(0,),
                dtype='unknown',
                notes=f"Error loading: {str(e)}"
            ))
    
    return results


def generate_report(results: List[ValidationResult]) -> ValidationReport:
    """Generate summary report from validation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    max_errors = [r.max_absolute_error for r in results if r.passed]
    
    summary = {
        'total_tests': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / total if total > 0 else 0,
        'overall_max_absolute_error': max(max_errors) if max_errors else float('inf'),
        'status': 'PASSED' if failed == 0 else 'FAILED'
    }
    
    return ValidationReport(results=results, summary=summary)


def print_report(report: ValidationReport):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("NUMERICAL VALIDATION REPORT")
    print("=" * 70)
    
    for r in report.results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n{status} {r.name}")
        print(f"  Shape: {r.shape}, dtype: {r.dtype}")
        print(f"  Max absolute error: {r.max_absolute_error:.2e}")
        print(f"  Max relative error: {r.max_relative_error:.2e}")
        print(f"  RMS error: {r.rms_error:.2e}")
        print(f"  Tolerances: rtol={r.rtol:.0e}, atol={r.atol:.0e}")
        if r.notes:
            print(f"  Notes: {r.notes}")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    s = report.summary
    print(f"Total: {s['total_tests']}, Passed: {s['passed']}, Failed: {s['failed']}")
    print(f"Pass rate: {s['pass_rate']*100:.1f}%")
    print(f"Overall max error: {s['overall_max_absolute_error']:.2e}")
    print(f"\nStatus: {s['status']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate numerical results against reference"
    )
    parser.add_argument(
        "--jax", "-j",
        type=Path,
        help="Path to JAX/Python result file"
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        help="Path to reference file"
    )
    parser.add_argument(
        "--jax-dir",
        type=Path,
        help="Directory with JAX result files"
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        help="Directory with reference files"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance (default: 1e-10)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance (default: 1e-12)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON report file"
    )
    parser.add_argument(
        "--pattern",
        default="*.npy",
        help="File pattern for directory mode (default: *.npy)"
    )
    
    args = parser.parse_args()
    
    # Validate single file or directory
    if args.jax and args.reference:
        results = [validate_files(args.jax, args.reference, args.rtol, args.atol)]
    elif args.jax_dir and args.ref_dir:
        results = validate_directories(
            args.jax_dir, args.ref_dir, args.rtol, args.atol, args.pattern
        )
    else:
        parser.print_help()
        print("\nError: Provide either --jax/--reference or --jax-dir/--ref-dir")
        sys.exit(1)
    
    # Generate and print report
    report = generate_report(results)
    print_report(report)
    
    # Save JSON report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with error code if failed
    sys.exit(0 if report.summary['status'] == 'PASSED' else 1)


if __name__ == "__main__":
    main()
