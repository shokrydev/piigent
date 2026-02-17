#!/usr/bin/env python3
"""
Demo script for running the PIIgent Weakness Analyzer.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.core.adversarial_orchestrator import run_adversarial_simulation

def main():
    print("Running PIIgent Adversarial Simulation (Red-Blue Team)...")
    print("Note: This performs an iterative Red Team attack and Blue Team defense.")
    
    try:
        report = run_adversarial_simulation(
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
