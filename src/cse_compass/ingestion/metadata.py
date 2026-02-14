from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Optional

# Company folder: "ACCESS ENGINEERING PLC (AEL.N0000)" -> ticker AEL.N0000
TICKER_FROM_FOLDER_RE = re.compile(r"\(([A-Z0-9]+\.[A-Z0-9]+)\)\s*$")
# Report folder: "371_1723200115194.06.2024" or "1141_1724759174228"
PERIOD_FROM_FOLDER_RE = re.compile(r"\.(\d{1,2})\.(\d{4})$")


def infer_metadata_from_processed_path(report_dir: Path) -> Dict[str, Any]:
    """
    Extract ticker, report_type, company_name, year from processed report path.
    Path: processed_reports/{Company Name (TICKER)}/{annual|quarterly|press_release}/{report_id}/
    Omits None values so Chroma (which rejects None in metadata) gets clean dicts.
    """
    md: Dict[str, Any] = {}
    parts = report_dir.parts
    # report_dir is .../Company (TICKER)/annual|quarterly|report_id
    if len(parts) >= 2:
        company_folder = parts[-3]
        report_type_folder = parts[-2]
        report_id = parts[-1]

        rt = report_type_folder.lower() if report_type_folder in (
            "annual", "quarterly", "press_release"
        ) else "unknown"
        md["report_type"] = rt

        m = TICKER_FROM_FOLDER_RE.search(company_folder)
        if m:
            md["ticker"] = m.group(1)
        md["company_name"] = company_folder

        pm = PERIOD_FROM_FOLDER_RE.search(report_id)
        if pm:
            month, year = int(pm.group(1)), int(pm.group(2))
            md["year"] = year
            md["period"] = f"{month:02d}.{year}"

    md["doc_id"] = report_dir.name
    # Chroma rejects None - only include defined values
    return {k: v for k, v in md.items() if v is not None}


FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]{2,10})_(?P<year>\d{4})_(?P<type>annual|q[1-4]|quarterly|interim)\.pdf$",
    re.IGNORECASE,
)

def infer_metadata_from_path(pdf_path: Path) -> Dict[str, Any]:
    """
    Extracts ticker/year/report_type from filename. Falls back safely if not matched.
    """
    md: Dict[str, Any] = {
        "ticker": None,
        "year": None,
        "report_type": None,
    }

    m = FILENAME_RE.match(pdf_path.name)
    if m:
        md["ticker"] = m.group("ticker").upper()
        md["year"] = int(m.group("year"))
        raw_type = m.group("type").lower()
        md["report_type"] = "annual" if raw_type == "annual" else raw_type
    else:
        # Fallback: use parent folder names if present (e.g., .../JKH/2023/annual.pdf)
        parts = [x.upper() for x in pdf_path.parts]
        # Very light heuristics
        for part in parts:
            if re.fullmatch(r"\d{4}", part):
                md["year"] = int(part)
        # ticker guess: first all-caps token in stem
        stem_tokens = re.split(r"[_\-\s]+", pdf_path.stem.upper())
        for tok in stem_tokens:
            if re.fullmatch(r"[A-Z0-9]{2,10}", tok):
                md["ticker"] = tok
                break

        # report type guess
        lower = pdf_path.stem.lower()
        if "annual" in lower:
            md["report_type"] = "annual"
        elif "q1" in lower:
            md["report_type"] = "q1"
        elif "q2" in lower:
            md["report_type"] = "q2"
        elif "q3" in lower:
            md["report_type"] = "q3"
        elif "q4" in lower:
            md["report_type"] = "q4"
        elif "quarter" in lower or "interim" in lower:
            md["report_type"] = "quarterly"

    # Always include a stable document_id (used for dedupe/versioning later)
    md["doc_id"] = f"{pdf_path.name}"
    return md
