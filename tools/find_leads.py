#!/usr/bin/env python3
"""tools/find_leads.py: GitHub Lead Discovery & Outreach Target Hunter for Chebyshev-Compressor.

Automates the discovery of open-source projects, IoT sensor implementations, and TinyML
repositories struggling with recurrent sequence modeling, RAM constraints, or TFLite Micro
arena overhead on resource-constrained microcontrollers (ESP32, STM32, Cortex-M).

Zero external dependencies (pure Python standard library).
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_SEARCH_PRESETS = [
    "esp32 lstm",
    "stm32 vibration anomaly",
    "tinyml time series",
    "tflite micro rnn",
    "microcontroller predictive maintenance",
    "edge ai sensor classification",
]


def get_github_token() -> Optional[str]:
    """Retrieves GitHub token from env or GitHub CLI (`gh auth token`)."""
    if "GITHUB_TOKEN" in os.environ and os.environ["GITHUB_TOKEN"].strip():
        return os.environ["GITHUB_TOKEN"].strip()
    if "GH_TOKEN" in os.environ and os.environ["GH_TOKEN"].strip():
        return os.environ["GH_TOKEN"].strip()
    
    # Try discovering via GitHub CLI if installed
    if shutil.which("gh"):
        try:
            res = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
    return None


def github_search_repositories(
    query: str,
    token: Optional[str] = None,
    sort: str = "updated",
    order: str = "desc",
    per_page: int = 15,
) -> List[Dict[str, Any]]:
    """Queries the GitHub Search API for repositories matching query."""
    encoded_q = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={encoded_q}&sort={sort}&order={order}&per_page={per_page}"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Chebyshev-Compressor-Lead-Hunter/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("items", [])
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"[!] GitHub API Rate Limit encountered for query '{query}'.", file=sys.stderr)
            print("    Tip: Set GITHUB_TOKEN environment variable or run 'gh auth login' to get 5,000 req/hr.", file=sys.stderr)
        else:
            print(f"[!] HTTP error {e.code} for query '{query}': {e.reason}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[!] Network or parsing error for query '{query}': {e}", file=sys.stderr)
        return []


def analyze_lead_angle(repo: Dict[str, Any], query: str) -> Dict[str, str]:
    """Generates customized value proposition and outreach angle based on repository metadata."""
    desc = (repo.get("description") or "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    name = repo.get("name", "").lower()
    all_text = f"{desc} {' '.join(topics)} {name}"

    angle = "General TinyML Recurrent Compression"
    pitch = (
        "Replaces heavy TFLite Micro tensor arenas (~32 KB) with exact C99 single-header "
        "inference and 96–448 byte persistent state RAM."
    )

    if any(k in all_text for k in ["vibration", "bearing", "motor", "accelerometer", "imu"]):
        angle = "Vibration & Predictive Maintenance (ESP32/STM32)"
        pitch = (
            "Check out our 96-byte persistent RAM C99 bearing vibration demo (examples/esp32_vibration/). "
            "Guarantees O(1) continuous state updates without FFT latency spikes or malloc."
        )
    elif any(k in all_text for k in ["ecg", "ppg", "health", "biomedical", "wearable"]):
        angle = "Wearable Biomedical Time-Series"
        pitch = (
            "Enables real-time streaming biosignal anomaly detection on sub-10KB RAM MCUs. "
            "Proven Lyapunov dissipation guarantees numerical stability over >10,000 steps without pole drift."
        )
    elif any(k in all_text for k in ["lstm", "gru", "rnn", "recurrent"]):
        angle = "Direct LSTM/GRU Teacher Replacement"
        pitch = (
            "Converts existing PyTorch/ONNX LSTM teachers into frozen Chebyshev state spaces. "
            "Maintains near-minimax continuous memory with 5.9 µs/step C99 execution and 100% C/PyTorch parity."
        )
    elif any(k in all_text for k in ["tflite", "arena", "memory", "ram", "oom"]):
        angle = "RAM & Arena Footprint Reduction"
        pitch = (
            "Eliminates 32 KB TFLM arena overhead entirely. Caller-allocated 448 B .bss state + 160 B stack scratch "
            "fits comfortably on low-end 2 KB SRAM microcontrollers."
        )

    return {"angle": angle, "pitch": pitch}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover GitHub repositories, research projects, and IoT sensor developers for Chebyshev-Compressor commercial outreach."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Custom search query (e.g. 'esp32 vibration anomaly' or 'tinyml lstm'). If omitted, runs default preset queries.",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=2,
        help="Minimum GitHub stars threshold (default: 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum unique leads to collect (default: 25).",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="docs/marketing/LEADS_REPORT.md",
        help="Path to save Markdown outreach report (default: docs/marketing/LEADS_REPORT.md).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="docs/marketing/leads.csv",
        help="Path to save CSV export (default: docs/marketing/leads.csv).",
    )
    args = parser.parse_args()

    token = get_github_token()
    token_status = "Authenticated (5,000 req/hr limit)" if token else "Unauthenticated (60 req/hr limit)"
    print("=" * 70)
    print("CHEBYSHEV-COMPRESSOR: GITHUB LEAD HUNTER & OUTREACH DISCOVERY")
    print("=" * 70)
    print(f"[*] API Status : {token_status}")
    
    queries = [args.query] if args.query else DEFAULT_SEARCH_PRESETS
    collected_leads: Dict[str, Dict[str, Any]] = {}

    for q in queries:
        if len(collected_leads) >= args.limit:
            break
        print(f"[*] Searching GitHub for: '{q}'...")
        items = github_search_repositories(q, token=token, per_page=15)
        
        for item in items:
            repo_id = item["full_name"]
            stars = item.get("stargazers_count", 0)
            
            # Filter out own repository and repositories below star threshold
            if "chebyshev-compressor" in repo_id.lower():
                continue
            if stars < args.min_stars:
                continue
                
            if repo_id not in collected_leads:
                analysis = analyze_lead_angle(item, q)
                collected_leads[repo_id] = {
                    "full_name": repo_id,
                    "url": item["html_url"],
                    "description": item.get("description") or "(No description)",
                    "stars": stars,
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language") or "C / C++",
                    "updated_at": item.get("updated_at", "")[:10],
                    "matched_query": q,
                    "angle": analysis["angle"],
                    "pitch": analysis["pitch"],
                    "issues_url": f"{item['html_url']}/issues",
                }
            if len(collected_leads) >= args.limit:
                break
        time.sleep(1.0)  # Rate pacing

    leads_list = list(collected_leads.values())
    leads_list.sort(key=lambda x: x["stars"], reverse=True)

    print(f"\n[+] Total high-value leads identified: {len(leads_list)}")
    if not leads_list:
        print("[!] No leads matched criteria. Try lowering --min-stars or checking network connection.")
        return 0

    # Print top 5 summary to console
    print("\n--- TOP DISCOVERED TARGETS ---")
    for i, lead in enumerate(leads_list[:5], start=1):
        print(f"[{i}] {lead['full_name']} (⭐ {lead['stars']} | {lead['language']})")
        print(f"    URL   : {lead['url']}")
        print(f"    Angle : {lead['angle']}")
        print(f"    Pitch : {lead['pitch']}")
        print()

    # Export to CSV
    csv_path = args.out_csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "full_name",
            "url",
            "stars",
            "forks",
            "language",
            "updated_at",
            "matched_query",
            "angle",
            "pitch",
            "issues_url",
            "description",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in leads_list:
            writer.writerow(row)
    print(f"[+] Saved structured CSV to: {csv_path}")

    # Export to Markdown
    md_path = args.out_md
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Commercial Outreach Leads: IoT Sensor & TinyML Repositories\n\n")
        f.write(f"Generated on: **{today_str}** via `tools/find_leads.py`.\n\n")
        f.write("These open-source repositories and engineering teams process time-series/sensor data on microcontrollers and represent ideal candidates for **Deployment Sprint ($79)** or **Research Package ($149)** productized services.\n\n")
        f.write("---\n\n")
        f.write("## 1. Top Identified Repositories\n\n")
        f.write("| Repository | Stars | Language | Dominant Angle | Action Link |\n")
        f.write("| :--- | ---: | :--- | :--- | :--- |\n")
        for row in leads_list:
            f.write(f"| [{row['full_name']}]({row['url']}) | ⭐ {row['stars']} | {row['language']} | {row['angle']} | [View Issues]({row['issues_url']}) |\n")
        
        f.write("\n---\n\n")
        f.write("## 2. Detailed Outreach Profiles & Customized Pitch Hooks\n\n")
        for i, row in enumerate(leads_list, start=1):
            f.write(f"### {i}. [{row['full_name']}]({row['url']})\n\n")
            f.write(f"- **Description**: {row['description']}\n")
            f.write(f"- **Last Updated**: `{row['updated_at']}` | **Stars**: ⭐ {row['stars']}\n")
            f.write(f"- **Target Angle**: **{row['angle']}**\n")
            f.write(f"- **Customized Outreach Pitch**:\n")
            f.write(f"  > *\"{row['pitch']}\"*\n")
            f.write(f"- **Contact Action**: [Open Discussion / Issue]({row['issues_url']})\n\n")

        f.write("---\n\n")
        f.write("## 3. High-Converting 3-Paragraph Outreach Template\n\n")
        f.write("When reaching out via GitHub Issue, Discussion, or LinkedIn/Email to the maintainers above, use this concise, non-salesy message:\n\n")
        f.write("```markdown\n")
        f.write("Hi [Maintainer Name / Team],\n\n")
        f.write("I came across [Repository Name] and really appreciated your work on embedded time-series processing. ")
        f.write("I noticed recurrent sensor models (like LSTM or TFLite Micro arenas) typically require 16–32 KB of SRAM on MCUs, which constrains sampling horizons.\n\n")
        f.write("We recently open-sourced Chebyshev-Compressor (https://github.com/Genesis410/chebyshev-compressor), which converts recurrent models into continuous Chebyshev state-space projections. ")
        f.write("On an ESP32, it achieves real-time 3-axis vibration monitoring in strictly 96 bytes of persistent RAM with zero dynamic malloc and full C99/PyTorch parity.\n\n")
        f.write("If you're interested in testing or benchmarking your model against an order-8 Chebyshev engine, feel free to run the open-source pipeline or let me know if you'd like a quick model conversion report.\n\n")
        f.write("Best regards,\n")
        f.write("Ihsan Akmal\n")
        f.write("ihsanakmal151@gmail.com\n")
        f.write("```\n")

    print(f"[+] Saved rich Markdown report to: {md_path}")
    print("=" * 70)
    print("SUCCESS: Lead discovery completed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
