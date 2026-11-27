#!/usr/bin/env python3
"""Probe public and user-supplied locations for the official BDD100K instance-mask ZIP.

The probe uses HTTP metadata and RemoteZip range requests; it does not download the
111 MB archive in full. A candidate is accepted only if the ZIP central directory
contains the official train/val instance bitmask structure.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from remotezip import RemoteZip

ARCHIVE = "bdd100k_ins_seg_labels_trainval.zip"
EXPECTED_MD5 = "651b41f229d7327d8c4af97772de4390"
OUT = "instance-source-probe"
os.makedirs(OUT, exist_ok=True)

CANDIDATES = [
    ("ETH official HTTPS", f"https://dl.cv.ethz.ch/bdd100k/data/{ARCHIVE}"),
    ("ETH official HTTP", f"http://dl.cv.ethz.ch/bdd100k/data/{ARCHIVE}"),
    ("Berkeley portal direct candidate", f"https://bdd-data.berkeley.edu/files/{ARCHIVE}"),
    ("Berkeley portal alternate candidate", f"https://bdd-data.berkeley.edu/{ARCHIVE}"),
    ("Berkeley S3 previously reported", f"https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/{ARCHIVE}"),
]

# A Drive URL/file ID can be supplied at dispatch time or as a repository secret.
drive_url = os.getenv("DRIVE_FILE_URL", "").strip()
if drive_url:
    CANDIDATES.append(("Google Drive user-supplied shared link", drive_url))

extra_url = os.getenv("BDD100K_INSTANCE_URL", "").strip()
if extra_url:
    CANDIDATES.append(("User-supplied dataset URL", extra_url))

def safe_url(url: str) -> str:
    p = urlparse(url)
    # Do not expose Drive confirmation tokens or other query parameters in logs.
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def inspect(label: str, url: str) -> dict:
    result = {"source": label, "url": safe_url(url), "status": "FAILED"}
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("URL must be a valid http(s) link")
        start = time.time()
        response = requests.get(
            url, headers={"Range": "bytes=0-15"}, allow_redirects=True,
            stream=True, timeout=(15, 30)
        )
        result["http_status"] = response.status_code
        result["final_host"] = urlparse(response.url).netloc
        result["content_type"] = response.headers.get("Content-Type", "")
        result["content_length"] = response.headers.get("Content-Length")
        result["content_range"] = response.headers.get("Content-Range", "")
        result["elapsed_seconds"] = round(time.time() - start, 2)
        first = response.raw.read(16)
        response.close()
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code}")
        if not (first.startswith(b"PK\\x03\\x04") or first.startswith(b"PK\\x05\\x06") or first.startswith(b"PK\\x07\\x08")):
            # Google Drive may return an HTML interstitial rather than file bytes.
            raise RuntimeError(f"Response is not ZIP data (prefix={first[:8]!r})")

        # RemoteZip reads the ZIP directory and selected metadata with byte ranges.
        with RemoteZip(url) as archive:
            names = archive.namelist()
        normalized = [n.replace("\\", "/").lower() for n in names]
        train = any("/labels/ins_seg/bitmasks/train/" in "/" + n for n in normalized)
        val = any("/labels/ins_seg/bitmasks/val/" in "/" + n for n in normalized)
        # Some mirrors omit the leading bdd100k directory; accept the canonical suffix.
        train = train or any("labels/ins_seg/bitmasks/train/" in n for n in normalized)
        val = val or any("labels/ins_seg/bitmasks/val/" in n for n in normalized)
        if not (train and val):
            raise RuntimeError(f"ZIP opened but official train/val bitmask folders missing (entries={len(names)})")
        result.update({
            "status": "PASS",
            "zip_entries": len(names),
            "train_bitmasks": train,
            "val_bitmasks": val,
            "md5_reference": EXPECTED_MD5,
            "note": "Valid instance-mask archive structure; full archive checksum not downloaded."
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:350]}"
    return result

results = []
for label, url in CANDIDATES:
    print(f"\n=== {label} ===", flush=True)
    row = inspect(label, url)
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if row["status"] == "PASS":
        # Save the first confirmed source for the training workflow.
        with open(os.path.join(OUT, "working-source.txt"), "w", encoding="utf-8") as f:
            f.write(url + "\n")
        print("CONFIRMED_SOURCE_FOUND", flush=True)
        break

summary = {
    "archive": ARCHIVE,
    "expected_official_md5": EXPECTED_MD5,
    "tested": len(results),
    "working_source": next((r["url"] for r in results if r["status"] == "PASS"), None),
    "results": results,
}
with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write("# BDD100K instance-mask source probe\n\n")
    f.write(f"- Archive: `{ARCHIVE}`\n- Official MD5 reference: `{EXPECTED_MD5}`\n")
    f.write(f"- Sources tested: {len(results)}\n\n")
    f.write("| Source | Status | HTTP | ZIP entries | Error |\n|---|---:|---:|---:|---|\n")
    for r in results:
        f.write(f"| {r['source']} | {r['status']} | {r.get('http_status','-')} | {r.get('zip_entries','-')} | {(r.get('error','') or '').replace('|','/')} |\n")
    if summary["working_source"]:
        f.write("\n## Confirmed source\n\nThe URL is saved in `working-source.txt`.\n")
    else:
        f.write("\nNo tested location returned the verified official train/val instance-mask archive.\n")

print("\n" + json.dumps({"working_source": summary["working_source"], "report": os.path.join(OUT, "report.json")}, ensure_ascii=False))
