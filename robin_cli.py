#!/usr/bin/env python3
import argparse
import json
import sys
import os
from datetime import datetime
import traceback
import contextlib

try:
    import config
    import search
    import scrape
    import llm
    import llm_utils
except ImportError as e:
    print(f"Initialization error: Could not import required modules ({e})", file=sys.stderr)
    sys.exit(1)

def save_investigation(data):
    """Saves the investigation data to a JSON file in the investigations directory."""
    os.makedirs("investigations", exist_ok=True)
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"investigation_{timestamp_str}.json"
    filepath = os.path.join("investigations", filename)

    save_data = {
        "timestamp": timestamp.isoformat(),
        "query": data["query"],
        "refined_query": data["refined_query"],
        "model": data["model"],
        "preset": data["preset"],
        "sources": data["sources"],
        "summary": data["summary"]
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    return filepath, save_data["timestamp"]

def main():
    parser = argparse.ArgumentParser(description="Robin Dark-Web OSINT CLI (Headless)")
    parser.add_argument("query", help="The initial investigation query")
    parser.add_argument("--preset", default="threat_intel", help="Investigation preset (default: threat_intel)")
    parser.add_argument("--max-results", type=int, default=None, help="Cap the number of search results before filtering")
    parser.add_argument("--json", action="store_true", help="Output only a single machine-readable JSON object")
    parser.add_argument("--quiet", action="store_true", help="Output only the saved JSON file path (overridden by --json)")
    parser.add_argument("--debug", action="store_true", help="Print full traceback on pipeline failure")

    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            # 1. Model resolution
            model_choices = llm_utils.get_model_choices()
            if not model_choices:
                print("No LLM models available. Configure at least one API key in /opt/robin/.env", file=sys.stderr)
                sys.exit(1)
            
            selected_model = model_choices[0]

            # 2. Preset validation
            if not hasattr(llm, "PRESET_PROMPTS"):
                print("Error: PRESET_PROMPTS not found in llm module.", file=sys.stderr)
                sys.exit(1)

            if args.preset not in llm.PRESET_PROMPTS:
                valid_presets = ", ".join(llm.PRESET_PROMPTS.keys())
                print(f"Unknown preset: {args.preset}. Valid: {valid_presets}", file=sys.stderr)
                sys.exit(1)

            # 3. LLM Initialization
            llm_instance = llm.get_llm(selected_model)
            
            # 4. Pipeline Execution
            refined_query = llm.refine_query(llm_instance, args.query)
            search_query = refined_query.replace(" ", "+")
            
            results = search.get_search_results(search_query)
            
            if args.max_results is not None:
                results = results[:args.max_results]
            
            filtered_results = llm.filter_results(llm_instance, refined_query, results)
            
            scraped_content = scrape.scrape_multiple(filtered_results)
            
            summary = llm.generate_summary(
                llm_instance, 
                args.query, 
                scraped_content, 
                preset=args.preset
            )

            # 5. Saving Results
            investigation_data = {
                "query": args.query,
                "refined_query": refined_query,
                "model": selected_model,
                "preset": args.preset,
                "sources": filtered_results,
                "summary": summary
            }

            filepath, timestamp_iso = save_investigation(investigation_data)

        # 6. Output Formatting
        if args.json:
            out_data = {
                "file": filepath,
                "timestamp": timestamp_iso,
                "query": args.query,
                "refined_query": refined_query,
                "model": selected_model,
                "preset": args.preset,
                "num_sources": len(filtered_results),
                "summary": summary
            }
            print(json.dumps(out_data))
        elif args.quiet:
            print(filepath)
        else:
            print(filepath)
            print(summary)

    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
