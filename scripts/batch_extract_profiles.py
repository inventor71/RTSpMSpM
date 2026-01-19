#!/usr/bin/env python3
"""
Batch extract timing metrics from multiple nsys profiling outputs.

This script processes the profiling directory structure:
  /home/RTSpMSpM/result/{dataset}/optixSpMSpM/
  /home/RTSpMSpM/result/{dataset}/cuSparse/

Usage:
    python3 batch_extract_profiles.py [result_dir] [output.csv]
    python3 batch_extract_profiles.py /home/RTSpMSpM/result results.csv
    python3 batch_extract_profiles.py  # Uses default result directory
"""

import sys
import os
import glob
from pathlib import Path
import csv
import sqlite3

# Default result directory
DEFAULT_RESULT_DIR = "/home/RTSpMSpM/result"

# Programs to look for
PROGRAMS = ["optixSpMSpM", "cuSparse"]


def ns_to_ms(nanoseconds):
    """Convert nanoseconds to milliseconds."""
    return nanoseconds / 1_000_000.0


def query_nvtx_range_time(cursor, range_name):
    """Query total time for a specific NVTX range."""
    query = """
    SELECT SUM(end - start) as total_ns
    FROM NVTX_EVENTS
    WHERE text = ?
    """
    cursor.execute(query, (range_name,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else 0


def query_memcpy_in_nvtx_range(cursor, nvtx_range_name):
    """Query GPU memcpy time that falls within a specific NVTX range."""
    query = """
    SELECT SUM(CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) as total_ns
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    JOIN NVTX_EVENTS ON (
        CUPTI_ACTIVITY_KIND_MEMCPY.start >= NVTX_EVENTS.start AND
        CUPTI_ACTIVITY_KIND_MEMCPY.end <= NVTX_EVENTS.end
    )
    WHERE NVTX_EVENTS.text = ?
    AND CUPTI_ACTIVITY_KIND_MEMCPY.copyKind IN (1, 2, 8)
    """
    cursor.execute(query, (nvtx_range_name,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else 0


def extract_optix_times(cursor):
    """Extract optixSpMSpM-specific timing metrics."""
    results = {}

    # Computation time (no I/O)
    comp_time_ns = query_nvtx_range_time(cursor, "computation time no io")
    results['computation_time_no_io_ns'] = comp_time_ns
    results['computation_time_no_io_ms'] = ns_to_ms(comp_time_ns)

    # storeSphereData GPU memcpy
    store_memcpy_ns = query_memcpy_in_nvtx_range(cursor, "storeSphereData")
    results['storeSphereData_gpu_memcpy_ns'] = store_memcpy_ns
    results['storeSphereData_gpu_memcpy_ms'] = ns_to_ms(store_memcpy_ns)

    # mat1ToGPU GPU memcpy
    mat1_memcpy_ns = query_memcpy_in_nvtx_range(cursor, "mat1ToGPU")
    results['mat1ToGPU_gpu_memcpy_ns'] = mat1_memcpy_ns
    results['mat1ToGPU_gpu_memcpy_ms'] = ns_to_ms(mat1_memcpy_ns)

    return results


def extract_cusparse_times(cursor):
    """Extract cuSparse-specific timing metrics."""
    results = {}

    # Computation time (no I/O)
    comp_time_ns = query_nvtx_range_time(cursor, "computation time no io")
    results['computation_time_no_io_ns'] = comp_time_ns
    results['computation_time_no_io_ms'] = ns_to_ms(comp_time_ns)

    return results


def extract_profile_times(sqlite_path, program_name):
    """Extract profiling times from nsys SQLite database."""

    if not os.path.exists(sqlite_path):
        print(f"  Error: File not found: {sqlite_path}", file=sys.stderr)
        return None

    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        # Check if required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        if 'NVTX_EVENTS' not in tables:
            print(f"  Warning: NVTX_EVENTS table not found in {sqlite_path}", file=sys.stderr)
            conn.close()
            return None

        # Extract program-specific metrics
        if "optix" in program_name.lower():
            results = extract_optix_times(cursor)
        else:
            results = extract_cusparse_times(cursor)

        conn.close()
        return results

    except sqlite3.Error as e:
        print(f"  Database error: {e}", file=sys.stderr)
        return None


def find_sqlite_files(directory):
    """Find all .sqlite files in a directory."""
    pattern = os.path.join(directory, "*.sqlite")
    return sorted(glob.glob(pattern))


def discover_datasets(result_dir):
    """Discover all datasets in the result directory."""
    datasets = []
    if not os.path.isdir(result_dir):
        return datasets

    for item in os.listdir(result_dir):
        item_path = os.path.join(result_dir, item)
        if os.path.isdir(item_path):
            # Check if it has optixSpMSpM or cuSparse subdirectories
            for program in PROGRAMS:
                prog_path = os.path.join(item_path, program)
                if os.path.isdir(prog_path):
                    datasets.append(item)
                    break

    return sorted(set(datasets))


def batch_process(result_dir, output_csv=None):
    """Process all profile files in the result directory structure."""

    datasets = discover_datasets(result_dir)

    if not datasets:
        print(f"No datasets found in: {result_dir}")
        return

    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}\n")

    # Collect all results
    all_results = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        for program in PROGRAMS:
            program_dir = os.path.join(result_dir, dataset, program)

            if not os.path.isdir(program_dir):
                print(f"  {program}: Directory not found")
                continue

            sqlite_files = find_sqlite_files(program_dir)

            if not sqlite_files:
                print(f"  {program}: No .sqlite files found")
                continue

            print(f"  {program}: Found {len(sqlite_files)} profile(s)")

            for sqlite_file in sqlite_files:
                filename = os.path.basename(sqlite_file)
                results = extract_profile_times(sqlite_file, program)

                if results:
                    all_results.append({
                        'dataset': dataset,
                        'program': program,
                        'filename': filename,
                        'filepath': sqlite_file,
                        **results
                    })
                    print(f"    ✓ {filename}: {results.get('computation_time_no_io_ms', 0):.3f} ms")
                else:
                    print(f"    ✗ {filename}: Failed to extract")

    # Print summary table
    print_summary(all_results)

    # Export to CSV if requested
    if output_csv:
        export_batch_csv(all_results, output_csv)


def print_summary(all_results):
    """Print a summary table of all results."""

    if not all_results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*100}")
    print("Summary: Computation Time (no I/O) in milliseconds")
    print(f"{'='*100}")
    print(f"{'Dataset':<25} {'Program':<15} {'Comp Time (ms)':<15} {'Profile':<40}")
    print(f"{'-'*100}")

    for result in all_results:
        comp_time = result.get('computation_time_no_io_ms', 0)
        print(f"{result['dataset']:<25} {result['program']:<15} {comp_time:>14.3f} {result['filename']:<40}")

    print(f"{'='*100}\n")

    # Print comparison by dataset
    print(f"\n{'='*80}")
    print("Comparison by Dataset (Average Computation Time)")
    print(f"{'='*80}")
    print(f"{'Dataset':<25} {'optixSpMSpM (ms)':<20} {'cuSparse (ms)':<20} {'Speedup':<15}")
    print(f"{'-'*80}")

    # Group by dataset
    from collections import defaultdict
    dataset_times = defaultdict(lambda: {'optixSpMSpM': [], 'cuSparse': []})

    for result in all_results:
        dataset = result['dataset']
        program = result['program']
        comp_time = result.get('computation_time_no_io_ms', 0)
        if comp_time > 0:
            dataset_times[dataset][program].append(comp_time)

    for dataset in sorted(dataset_times.keys()):
        times = dataset_times[dataset]
        optix_avg = sum(times['optixSpMSpM']) / len(times['optixSpMSpM']) if times['optixSpMSpM'] else 0
        cusparse_avg = sum(times['cuSparse']) / len(times['cuSparse']) if times['cuSparse'] else 0

        speedup = cusparse_avg / optix_avg if optix_avg > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"

        optix_str = f"{optix_avg:.3f}" if optix_avg > 0 else "N/A"
        cusparse_str = f"{cusparse_avg:.3f}" if cusparse_avg > 0 else "N/A"

        print(f"{dataset:<25} {optix_str:<20} {cusparse_str:<20} {speedup_str:<15}")

    print(f"{'='*80}\n")


def export_batch_csv(all_results, output_path):
    """Export batch results to CSV."""

    # Determine all unique fields
    all_fields = set()
    for result in all_results:
        all_fields.update(result.keys())

    # Define field order
    base_fields = ['dataset', 'program', 'filename', 'computation_time_no_io_ms', 'computation_time_no_io_ns']
    extra_fields = sorted([f for f in all_fields if f not in base_fields and f != 'filepath'])
    fieldnames = base_fields + extra_fields

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for result in all_results:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"Results exported to: {output_path}")


def main():
    # Parse arguments
    result_dir = DEFAULT_RESULT_DIR
    output_csv = None

    if len(sys.argv) >= 2:
        if sys.argv[1] in ['--help', '-h']:
            print(__doc__)
            sys.exit(0)
        result_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        output_csv = sys.argv[2]

    if not os.path.isdir(result_dir):
        print(f"Error: Directory not found: {result_dir}")
        sys.exit(1)

    print(f"Processing profiles from: {result_dir}")
    batch_process(result_dir, output_csv)


if __name__ == "__main__":
    main()
