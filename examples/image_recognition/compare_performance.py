#!/usr/bin/env python
"""
Performance Comparison: MiniTen vs PyTorch vs TensorFlow

This script runs image classification on all three frameworks
and generates a comparison report.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_comparison():
    """Run all frameworks and compare results."""
    
    print("="*70)
    print("FRAMEWORK PERFORMANCE COMPARISON")
    print("MiniTen vs PyTorch vs TensorFlow")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Run MiniTen
    print("Running MiniTen...")
    print("-"*70)
    try:
        from train_miniten import train_miniten
        miniten_result = train_miniten()
        results.append(miniten_result)
    except Exception as e:
        print(f"MiniTen failed: {e}")
        results.append({"framework": "MiniTen", "error": str(e)})
    print()
    
    # Run PyTorch
    print("Running PyTorch...")
    print("-"*70)
    try:
        from train_pytorch import train_pytorch
        pytorch_result = train_pytorch()
        if pytorch_result:
            results.append(pytorch_result)
        else:
            results.append({"framework": "PyTorch", "error": "Not installed"})
    except Exception as e:
        print(f"PyTorch failed: {e}")
        results.append({"framework": "PyTorch", "error": str(e)})
    print()
    
    # Run TensorFlow
    print("Running TensorFlow...")
    print("-"*70)
    try:
        from train_tensorflow import train_tensorflow
        tf_result = train_tensorflow()
        if tf_result:
            results.append(tf_result)
        else:
            results.append({"framework": "TensorFlow", "error": "Not installed"})
    except Exception as e:
        print(f"TensorFlow failed: {e}")
        results.append({"framework": "TensorFlow", "error": str(e)})
    print()
    
    # Generate comparison report
    generate_report(results)
    
    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to comparison_results.json")
    
    return results


def generate_report(results):
    """Generate a markdown comparison report."""
    
    print("="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Filter successful results
    successful = [r for r in results if "error" not in r]
    
    if not successful:
        print("No frameworks ran successfully!")
        return
    
    # Print table
    print(f"\n{'Framework':<15} {'Train Time':>12} {'Accuracy':>10} {'Latency':>12} {'Device':>10}")
    print("-"*70)
    
    for r in results:
        if "error" in r:
            print(f"{r['framework']:<15} {'Error: ' + r['error'][:40]:>}")
        else:
            train_time = f"{r.get('train_time', 0):.2f}s"
            accuracy = f"{r.get('final_accuracy', 0):.2%}"
            latency = f"{r.get('latency_ms', 0):.2f}ms"
            device = r.get('device', 'CPU')[:10]
            print(f"{r['framework']:<15} {train_time:>12} {accuracy:>10} {latency:>12} {device:>10}")
    
    print("-"*70)
    
    # Analysis
    print("\n📊 ANALYSIS")
    print("-"*40)
    
    if len(successful) > 1:
        # Find fastest training
        fastest_train = min(successful, key=lambda x: x.get('train_time', float('inf')))
        print(f"⚡ Fastest Training: {fastest_train['framework']} ({fastest_train['train_time']:.2f}s)")
        
        # Find lowest latency
        lowest_latency = min(successful, key=lambda x: x.get('latency_ms', float('inf')))
        print(f"🚀 Lowest Latency: {lowest_latency['framework']} ({lowest_latency['latency_ms']:.2f}ms)")
        
        # Find highest accuracy
        highest_acc = max(successful, key=lambda x: x.get('final_accuracy', 0))
        print(f"🎯 Highest Accuracy: {highest_acc['framework']} ({highest_acc['final_accuracy']:.2%})")
    
    # Notes
    print("\n📝 NOTES")
    print("-"*40)
    print("• All frameworks use identical model architecture (2-layer CNN)")
    print("• All frameworks use identical synthetic dataset")
    print("• Latency measured on batch of 32 samples")
    print("• Training for 10 epochs with SGD optimizer (lr=0.01)")
    
    # MiniTen specific notes
    miniten_results = [r for r in results if r['framework'] == 'MiniTen']
    if miniten_results and 'use_cython' in miniten_results[0]:
        if miniten_results[0]['use_cython']:
            print("• MiniTen: Using Cython-optimized layers")
        else:
            print("• MiniTen: Using NumPy fallback (Cython not built)")
    
    # Generate markdown report
    report_md = generate_markdown_report(results)
    with open("comparison_report.md", "w") as f:
        f.write(report_md)
    print("\nMarkdown report saved to comparison_report.md")


def generate_markdown_report(results):
    """Generate a markdown report."""
    
    lines = [
        "# Performance Comparison Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This report compares MiniTen with PyTorch and TensorFlow on a simple",
        "image classification task using identical model architecture and data.",
        "",
        "## Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Model | 2-layer CNN (Conv-Pool-Conv-Pool-FC) |",
        "| Dataset | Synthetic patterns (10 classes, 28x28) |",
        "| Training samples | 800 |",
        "| Test samples | 200 |",
        "| Batch size | 32 |",
        "| Epochs | 10 |",
        "| Optimizer | SGD (lr=0.01) |",
        "",
        "## Results",
        "",
        "| Framework | Train Time | Accuracy | Latency (32) | Device |",
        "|-----------|------------|----------|--------------|--------|",
    ]
    
    for r in results:
        if "error" in r:
            lines.append(f"| {r['framework']} | Error | - | - | - |")
        else:
            lines.append(
                f"| {r['framework']} | "
                f"{r.get('train_time', 0):.2f}s | "
                f"{r.get('final_accuracy', 0):.2%} | "
                f"{r.get('latency_ms', 0):.2f}ms | "
                f"{r.get('device', 'CPU')} |"
            )
    
    lines.extend([
        "",
        "## Analysis",
        "",
    ])
    
    successful = [r for r in results if "error" not in r]
    
    if len(successful) > 1:
        fastest = min(successful, key=lambda x: x.get('train_time', float('inf')))
        lines.append(f"- **Fastest Training**: {fastest['framework']} ({fastest['train_time']:.2f}s)")
        
        lowest = min(successful, key=lambda x: x.get('latency_ms', float('inf')))
        lines.append(f"- **Lowest Latency**: {lowest['framework']} ({lowest['latency_ms']:.2f}ms)")
        
        best_acc = max(successful, key=lambda x: x.get('final_accuracy', 0))
        lines.append(f"- **Highest Accuracy**: {best_acc['framework']} ({best_acc['final_accuracy']:.2%})")
    
    lines.extend([
        "",
        "## Conclusions",
        "",
        "### MiniTen Advantages",
        "",
        "1. **Minimal Dependencies**: Only requires NumPy and Cython",
        "2. **Small Footprint**: Ideal for edge devices with limited storage",
        "3. **Educational**: Clear, readable implementation",
        "4. **Customizable**: Easy to extend and modify",
        "",
        "### When to Use Each Framework",
        "",
        "| Use Case | Recommended |",
        "|----------|-------------|",
        "| Edge/IoT deployment | MiniTen |",
        "| Research/Prototyping | PyTorch |",
        "| Production at scale | TensorFlow |",
        "| Learning ML internals | MiniTen |",
        "",
        "---",
        "",
        "*Report generated by MiniTen comparison tool*",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_comparison()
