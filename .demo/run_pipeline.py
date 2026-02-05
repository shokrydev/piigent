#!/usr/bin/env python3
"""
Demo script for running the PIIgent privacy pipeline.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.pipeline_graph import run_pipeline

def main():
    print("Running PIIgent Privacy Pipeline...")
    
    document = """
    Entlassungsbrief - Charité Berlin
    Patient: Max Mustermann
    KVNR: A123456780
    PLZ: 10117 Berlin
    """
    
    print(f"Input Document:\n{document}\n")
    
    try:
        result = run_pipeline(
            document=document,
            confidence_threshold=0.7,
            human_in_loop=False,
            preset='clinical',
        )
        
        print("-" * 50)
        print("Anonymized Output:")
        print(result['anonymized_text'])
        print("-" * 50)
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("Note: Ensure dependencies are installed and specific models (like Ollama) are available if used.")

if __name__ == "__main__":
    main()
