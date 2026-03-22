import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any


def find_pareto_frontier(points: List[Dict[str, Any]], x_metric: str, y_metric: str, x_higher_is_better: bool = True,
                         y_higher_is_better: bool = False):
    """
    Finds the Pareto frontier for a set of points.
    
    Default: x=throughput (maximize), y=latency (minimize).
    """
    if not points:
        return []

    # Sort points primarily by x_metric
    sorted_points = sorted(points, key=lambda p: p[x_metric], reverse=x_higher_is_better)

    frontier = []
    if not sorted_points:
        return frontier

    current_best_y = sorted_points[0][y_metric]
    frontier.append(sorted_points[0])

    for i in range(1, len(sorted_points)):
        point = sorted_points[i]
        is_better_y = point[y_metric] < current_best_y if not y_higher_is_better else point[y_metric] > current_best_y

        if is_better_y:
            frontier.append(point)
            current_best_y = point[y_metric]

    return frontier


def process_results(input_dir: str, output_file: str = None):
    """Loads all benchmark JSONs in a directory and computes Pareto frontier."""
    results = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        data = json.load(f)
                        if "summary" in data:
                            results.append(data["summary"])
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    if not results:
        print("No results found to analyze.")
        return

    # Throughput vs p95 Latency
    frontier = find_pareto_frontier(results, "throughput_rps", "latency_p95_ms")

    # Create Markdown Summary
    md_output = "## Pareto Frontier Analysis (Throughput vs P95 Latency)\n\n"
    md_output += "| Throughput (RPS) | P95 Latency (ms) | Tokens/Sec | Concurrency | Config |\n"
    md_output += "| :--- | :--- | :--- | :--- | :--- |\n"

    for p in frontier:
        config_str = f"Tier: {p.get('tier', 'N/A')}"
        md_output += (f"| {p['throughput_rps']:.2f} | {p['latency_p95_ms']:.2f} | {p['throughput_tps']:.2f} | "
                      f"{p['concurrency']} | {config_str} |\n")

    print(md_output)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(md_output)
        print(f"Summary written to {output_file}")

        # Also write CSV of all results
        df = pd.DataFrame(results)
        csv_file = output_file.replace(".md", ".csv")
        df.to_csv(csv_file, index=False)
        print(f"All results written to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Benchmark Pareto Analysis")
    parser.add_argument("--input-dir", required=True, help="Directory containing benchmark JSON results")
    parser.add_argument("--output", help="Output Markdown file for the summary")

    args = parser.parse_args()
    process_results(args.input_dir, args.output)
