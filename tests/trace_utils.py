# Copyright (C) 2026 SPE Team - Dev 1
"""
Trace utilities for formatting and printing verification trace tables during test execution.

Supports toggling via the SHOW_TRACE environment variable (defaults to '1' / enabled).
Set SHOW_TRACE=0 to run tests silently.
"""

import os
from typing import Any, List, Optional


def should_show_trace() -> bool:
    """Check if trace table printing is enabled via environment variable SHOW_TRACE."""
    val = os.environ.get("SHOW_TRACE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def format_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> str:
    """
    Format data as a clean ASCII verification table.

    Args:
        title: Table title or description.
        headers: List of column header strings.
        rows: List of row data (each row is a list of cell values).
        alignments: Optional list of 'left', 'right', or 'center' per column.
    """
    col_count = len(headers)
    str_rows = [[str(cell) for cell in row] for row in rows]

    # Calculate column widths based on headers and cells
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx in range(min(col_count, len(row))):
            widths[idx] = max(widths[idx], len(row[idx]))

    if alignments is None:
        alignments = ["left"] * col_count

    def align_cell(text: str, width: int, align: str) -> str:
        if align == "right":
            return text.rjust(width)
        elif align == "center":
            return text.center(width)
        return text.ljust(width)

    # Build border strings
    sep_line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = (
        "| "
        + " | ".join(align_cell(headers[i], widths[i], "center") for i in range(col_count))
        + " |"
    )

    lines = []
    lines.append(f"\n=== {title} ===")
    lines.append(sep_line)
    lines.append(header_line)
    lines.append(sep_line)
    for row in str_rows:
        row_line = (
            "| "
            + " | ".join(
                align_cell(row[i] if i < len(row) else "", widths[i], alignments[i])
                for i in range(col_count)
            )
            + " |"
        )
        lines.append(row_line)
    lines.append(sep_line)
    return "\n".join(lines)


def print_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> None:
    """Print the formatted table to stdout if SHOW_TRACE is enabled."""
    if should_show_trace():
        import sys
        sys.stdout.write(format_table(title, headers, rows, alignments) + "\n\n")
        sys.stdout.flush()
