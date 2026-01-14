"""
HTML Dashboard for MiniTen training monitor.

Generates interactive HTML dashboards for visualizing training progress.
"""

import os
import json
from pathlib import Path
from datetime import datetime


class Dashboard:
    """
    Generate HTML dashboards for training visualization.
    
    Args:
        experiments_dir: Directory containing experiment logs
        
    Example:
        >>> dashboard = Dashboard("./runs")
        >>> dashboard.generate("report.html")
    """
    
    def __init__(self, experiments_dir="./runs"):
        """Initialize dashboard generator."""
        self.experiments_dir = Path(experiments_dir)
    
    def generate(self, output_path, runs=None, title="MiniTen Training Dashboard"):
        """
        Generate an HTML dashboard.
        
        Args:
            output_path: Path to save HTML file
            runs: List of run names to include (all if None)
            title: Dashboard title
        """
        if runs is None:
            runs = self._get_all_runs()
        
        # Collect data
        runs_data = []
        for run_name in runs:
            run_data = self._load_run_data(run_name)
            if run_data:
                runs_data.append(run_data)
        
        # Generate HTML
        html = self._generate_html(runs_data, title)
        
        # Save
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Dashboard saved to: {output_path}")
    
    def _get_all_runs(self):
        """Get all available runs."""
        if not self.experiments_dir.exists():
            return []
        
        runs = []
        for d in self.experiments_dir.iterdir():
            if d.is_dir() and (d / "scalars.jsonl").exists():
                runs.append(d.name)
        return sorted(runs)
    
    def _load_run_data(self, run_name):
        """Load data for a single run."""
        run_dir = self.experiments_dir / run_name
        
        # Load metadata
        metadata = {}
        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Load scalars
        metrics = {}
        scalars_file = run_dir / "scalars.jsonl"
        if scalars_file.exists():
            with open(scalars_file) as f:
                for line in f:
                    entry = json.loads(line)
                    name = entry["name"]
                    if name not in metrics:
                        metrics[name] = {"steps": [], "values": []}
                    metrics[name]["steps"].append(entry["step"])
                    metrics[name]["values"].append(entry["value"])
        
        # Load hyperparameters
        hparams = {}
        hparams_file = run_dir / "hparams.json"
        if hparams_file.exists():
            with open(hparams_file) as f:
                hparams = json.load(f)
        
        return {
            "name": run_name,
            "metadata": metadata,
            "metrics": metrics,
            "hparams": hparams,
        }
    
    def _generate_html(self, runs_data, title):
        """Generate HTML content."""
        # Generate metrics charts data
        charts_data = {}
        for run in runs_data:
            for metric_name, values in run["metrics"].items():
                if metric_name not in charts_data:
                    charts_data[metric_name] = []
                charts_data[metric_name].append({
                    "name": run["name"],
                    "steps": values["steps"],
                    "values": values["values"],
                })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" integrity="sha384-/FhHKJKLyxN8HQyJVBV+KknCQZBLKPLDSRGIcxNdRXlUxNmjpNXoXlZTMTz1k3bR" crossorigin="anonymous"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-container h2 {{
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 18px;
        }}
        .runs-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .runs-table th, .runs-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .runs-table th {{
            background: #2c3e50;
            color: white;
        }}
        .runs-table tr:hover {{
            background: #f9f9f9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{title}</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Runs</h3>
                <div class="value">{len(runs_data)}</div>
            </div>
            <div class="summary-card">
                <h3>Metrics Tracked</h3>
                <div class="value">{len(charts_data)}</div>
            </div>
            <div class="summary-card">
                <h3>Generated</h3>
                <div class="value">{datetime.now().strftime("%Y-%m-%d")}</div>
            </div>
        </div>
        
        <div class="charts">
"""
        
        # Add charts
        colors = [
            'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
            'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        ]
        
        for i, (metric_name, runs) in enumerate(charts_data.items()):
            chart_id = f"chart_{i}"
            html += f"""
            <div class="chart-container">
                <h2>{metric_name}</h2>
                <canvas id="{chart_id}"></canvas>
            </div>
"""
        
        html += """
        </div>
        
        <h2 style="margin: 30px 0 15px;">Runs Overview</h2>
        <table class="runs-table">
            <thead>
                <tr>
                    <th>Run Name</th>
                    <th>Start Time</th>
                    <th>Hyperparameters</th>
                    <th>Final Metrics</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add runs table rows
        for run in runs_data:
            start_time = run["metadata"].get("start_time", "N/A")
            
            # Format hyperparameters
            hparams_str = ", ".join(
                f"{k}: {v}" for k, v in list(run["hparams"].items())[:3]
            ) or "N/A"
            
            # Get final metrics
            final_metrics = []
            for metric_name, values in list(run["metrics"].items())[:3]:
                if values["values"]:
                    final_metrics.append(f"{metric_name}: {values['values'][-1]:.4f}")
            final_metrics_str = ", ".join(final_metrics) or "N/A"
            
            html += f"""
                <tr>
                    <td>{run["name"]}</td>
                    <td class="timestamp">{start_time}</td>
                    <td>{hparams_str}</td>
                    <td>{final_metrics_str}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <script>
"""
        
        # Add chart JavaScript
        for i, (metric_name, runs) in enumerate(charts_data.items()):
            chart_id = f"chart_{i}"
            datasets_js = []
            for j, run in enumerate(runs):
                color = colors[j % len(colors)]
                datasets_js.append(f"""{{
                    label: '{run["name"]}',
                    data: {run["values"]},
                    borderColor: '{color}',
                    fill: false,
                    tension: 0.1
                }}""")
            
            # Use the longest step array for labels
            max_steps = max(len(run["steps"]) for run in runs) if runs else 0
            labels = list(range(max_steps))
            
            html += f"""
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{', '.join(datasets_js)}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: false }},
                    x: {{ title: {{ display: true, text: 'Step' }} }}
                }}
            }}
        }});
"""
        
        html += """
    </script>
</body>
</html>
"""
        
        return html
    
    def live_dashboard(self, run_name, port=8080):
        """
        Start a live updating dashboard server.
        
        Args:
            run_name: Run to monitor
            port: Server port
        """
        # Placeholder for live dashboard functionality
        print(f"Live dashboard would start on port {port} for run: {run_name}")
        print("This feature requires additional web server dependencies.")
