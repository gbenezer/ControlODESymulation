#!/usr/bin/env python3
"""
Test that ALL Quarto documentation executes successfully.

This is critical since all docs are now executable.
"""

import subprocess
from pathlib import Path
import sys
import time


def test_quarto_file(qmd_file: Path) -> tuple[bool, float, str]:
    """
    Test that a single .qmd file renders without errors.

    Returns:
        (success, render_time, error_message)
    """

    print(f"Testing: {qmd_file.name}...", end=" ", flush=True)

    start = time.time()
    try:
        result = subprocess.run(
            ["quarto", "render", str(qmd_file), "--execute"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per file
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✓ PASS ({elapsed:.1f}s)")
            return True, elapsed, ""
        else:
            print(f"✗ FAIL ({elapsed:.1f}s)")
            error_lines = result.stderr.split("\n")
            # Get last 20 lines of error
            error_msg = "\n".join(error_lines[-20:])
            print(f"  Error preview: {error_lines[-3:]}")
            return False, elapsed, error_msg

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"✗ TIMEOUT (>{elapsed:.0f}s)")
        return False, elapsed, "Timeout exceeded"
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ ERROR: {e}")
        return False, elapsed, str(e)


def main():
    """Test all Quarto documentation files."""

    docs_root = Path("docs")

    # Find all .qmd files
    qmd_files = list(docs_root.rglob("*.qmd"))
    # Exclude index files from listings
    qmd_files = [f for f in qmd_files if f.name != "_listing.qmd"]

    print("=" * 70)
    print(f"Testing ALL {len(qmd_files)} Quarto Documentation Files")
    print("Full Execution Mode - Every Code Block Must Run")
    print("=" * 70)
    print()

    results = []
    total_time = 0

    for qmd_file in sorted(qmd_files):
        passed, elapsed, error = test_quarto_file(qmd_file)
        results.append((qmd_file, passed, elapsed, error))
        total_time += elapsed

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed_count = sum(1 for _, p, _, _ in results if p)
    failed_count = len(results) - passed_count

    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Failed: {failed_count}/{len(results)}")
    print(f"Total render time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average per file: {total_time/len(results):.1f}s")

    if failed_count > 0:
        print("\n" + "=" * 70)
        print("Failed Files (Details)")
        print("=" * 70)
        for qmd_file, passed, elapsed, error in results:
            if not passed:
                print(f"\n{qmd_file.relative_to(docs_root)}:")
                print(f"  Time: {elapsed:.1f}s")
                if error:
                    print(f"  Error:\n{error}")

        print("\n" + "=" * 70)
        sys.exit(1)
    else:
        print("\n✓ All documentation files executed successfully!")
        print(f"✓ Total of {sum(r[2] for r in results)/60:.1f} minutes of computation verified")
        sys.exit(0)


if __name__ == "__main__":
    main()
