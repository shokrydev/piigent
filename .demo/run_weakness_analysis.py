#!/usr/bin/env python3
"""
Demo script for running the PIIgent Weakness Analyzer.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.weakness_analyzer import analyze_weaknesses

def main():
    print("Running PIIgent Weakness Analyzer...")
    print("Note: This performs an iterative analysis which may take some time.")
    
    try:
        report = analyze_weaknesses(
            num_docs=5,  # Reduced for demo purposes
            max_rounds=2,
            enable_evolution=True, 
        )
        
        print("\nAnalysis Complete.")
        print("-" * 50)
        print("Recommendations:")
        for rec in report.get("recommendations", []):
            print(f"- {rec}")
            
        evolution = report.get("evolution", {})
        if evolution:
            print("\nPrompt Evolution:")
            print(f"Mutations Applied: {evolution.get('mutations_applied', 0)}")
            print(f"Evolved Genome ID: {evolution.get('evolved_genome_id')}")
            
    except Exception as e:
        print(f"Error running weakness analysis: {e}")

if __name__ == "__main__":
    main()
