#!/usr/bin/env python3
"""
Extract specific timing metrics from nsys profiling output.

This script extracts:
1. "computation time no io" NVTX range time
2. GPU memcpy times during "storeSphereData"
3. GPU memcpy times during "mat1ToGPU"

Usage:
    python3 extract_profile_times.py <profile.sqlite>
    python3 extract_profile_times.py wiki-Vote_profile.sqlite
"""

import sqlite3
import sys
import os
from pathlib import Path


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
    # copyKind: 1=HtoD, 2=DtoH, 8=DtoD

    cursor.execute(query, (nvtx_range_name,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else 0


def ns_to_ms(nanoseconds):
    """Convert nanoseconds to milliseconds."""
    return nanoseconds / 1_000_000.0


def extract_profile_times(sqlite_path):
    """Extract profiling times from nsys SQLite database."""

    if not os.path.exists(sqlite_path):
        print(f"Error: File not found: {sqlite_path}", file=sys.stderr)
        return None

    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Check if required tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    if 'NVTX_EVENTS' not in tables:
        print("Error: NVTX_EVENTS table not found. Make sure the profile was captured with NVTX enabled.", file=sys.stderr)
        conn.close()
        return None

    if 'CUPTI_ACTIVITY_KIND_MEMCPY' not in tables:
        print("Warning: CUPTI_ACTIVITY_KIND_MEMCPY table not found. GPU memcpy times will be 0.", file=sys.stderr)

    try:
        # Extract times
        results = {}

        # 1. Computation time (no I/O)
        comp_time_ns = query_nvtx_range_time(cursor, "computation time no io")
        results['computation_time_no_io_ns'] = comp_time_ns
        results['computation_time_no_io_ms'] = ns_to_ms(comp_time_ns)

        # 2. storeSphereData GPU memcpy
        store_memcpy_ns = query_memcpy_in_nvtx_range(cursor, "storeSphereData")
        results['storeSphereData_gpu_memcpy_ns'] = store_memcpy_ns
        results['storeSphereData_gpu_memcpy_ms'] = ns_to_ms(store_memcpy_ns)

        # 3. mat1ToGPU GPU memcpy
        mat1_memcpy_ns = query_memcpy_in_nvtx_range(cursor, "mat1ToGPU")
        results['mat1ToGPU_gpu_memcpy_ns'] = mat1_memcpy_ns
        results['mat1ToGPU_gpu_memcpy_ms'] = ns_to_ms(mat1_memcpy_ns)

        # 4. Total of requested times
        total_ns = comp_time_ns + store_memcpy_ns + mat1_memcpy_ns
        results['total_ns'] = total_ns
        results['total_ms'] = ns_to_ms(total_ns)

        # Also get some context - total NVTX time
        cursor.execute("SELECT SUM(end - start) FROM NVTX_EVENTS")
        total_nvtx_ns = cursor.fetchone()[0] or 0
        results['total_nvtx_time_ns'] = total_nvtx_ns
        results['total_nvtx_time_ms'] = ns_to_ms(total_nvtx_ns)

        conn.close()
        return results

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        conn.close()
        return None


def print_results(results, sqlite_path):
    """Print results in a formatted table."""

    print(f"\n{'='*70}")
    print(f"Profile Timing Extraction")
    print(f"{'='*70}")
    print(f"Source: {sqlite_path}")
    print(f"{'-'*70}")

    print(f"\n{'Metric':<40} {'Time (ms)':<15} {'Time (ns)':<15}")
    print(f"{'-'*70}")

    print(f"{'computation time no io':<40} {results['computation_time_no_io_ms']:>14.6f} {results['computation_time_no_io_ns']:>14}")
    print(f"{'storeSphereData (GPU memcpy)':<40} {results['storeSphereData_gpu_memcpy_ms']:>14.6f} {results['storeSphereData_gpu_memcpy_ns']:>14}")
    print(f"{'mat1ToGPU (GPU memcpy)':<40} {results['mat1ToGPU_gpu_memcpy_ms']:>14.6f} {results['mat1ToGPU_gpu_memcpy_ns']:>14}")
    print(f"{'-'*70}")
    print(f"{'TOTAL (requested metrics)':<40} {results['total_ms']:>14.6f} {results['total_ns']:>14}")
    print(f"{'-'*70}")
    print(f"{'Total NVTX time (all ranges)':<40} {results['total_nvtx_time_ms']:>14.6f} {results['total_nvtx_time_ns']:>14}")
    print(f"{'='*70}\n")


def export_csv(results, output_path):
    """Export results to CSV format."""

    with open(output_path, 'w') as f:
        f.write("metric,time_ms,time_ns\n")
        f.write(f"computation_time_no_io,{results['computation_time_no_io_ms']},{results['computation_time_no_io_ns']}\n")
        f.write(f"storeSphereData_gpu_memcpy,{results['storeSphereData_gpu_memcpy_ms']},{results['storeSphereData_gpu_memcpy_ns']}\n")
        f.write(f"mat1ToGPU_gpu_memcpy,{results['mat1ToGPU_gpu_memcpy_ms']},{results['mat1ToGPU_gpu_memcpy_ns']}\n")
        f.write(f"total,{results['total_ms']},{results['total_ns']}\n")

    print(f"Results exported to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_profile_times.py <profile.sqlite> [--csv output.csv]")
        print("\nExample:")
        print("  python3 extract_profile_times.py wiki-Vote_profile.sqlite")
        print("  python3 extract_profile_times.py wiki-Vote_profile.sqlite --csv results.csv")
        sys.exit(1)

    sqlite_path = sys.argv[1]

    # Extract times
    results = extract_profile_times(sqlite_path)

    if results is None:
        sys.exit(1)

    # Print results
    print_results(results, sqlite_path)

    # Export to CSV if requested
    if '--csv' in sys.argv:
        csv_index = sys.argv.index('--csv')
        if csv_index + 1 < len(sys.argv):
            csv_path = sys.argv[csv_index + 1]
            export_csv(results, csv_path)
        else:
            print("Error: --csv requires an output filename", file=sys.stderr)
            sys.exit(1)

    # Return 0 for success
    return 0


if __name__ == "__main__":
    sys.exit(main())
