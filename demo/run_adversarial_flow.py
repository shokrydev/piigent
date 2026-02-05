#!/usr/bin/env python3
"""
Adversarial demo script for PIIgent.
Tests the system on a "Hard Mode" clinical document without explicit labels 
and with intentional ambiguity.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.privacy_flow import run_flow

def main():
    print("=" * 60)
    print("PIIgent ADVERSARIAL FLOW DEMO (Hard Mode)")
    print("=" * 60)
    print("\nDESCRIPTION:")
    print("This demo uses a narrative clinical note without hints like 'Patient:' or 'KVNR:'.")
    print("It includes overlapping entities and context-dependent PII.\n")
    
    document = """
Arztbrief für Schmidt, Thomas. Der 45-jährige wurde am 12.01.2024 (A123456780) in der Station 4b aufgenommen. 
Er klagte über starke Schmerzen im Oberbauch. Schmidt wohnt in der Nähe von 10115 (Berlin Mitte), 
seine Telefonnummer ist 030-20930. 

Als Kontaktperson wurde seine Tochter Julia hinterlegt (j.schmidt@berlin-klinik.de). 
Dr. Wagner hat die Medikation für den Patienten angepasst. Die Entlassung ist für nächsten Montag geplant.
"""
    
    print("-" * 60)
    print("INPUT DOCUMENT (Adversarial):")
    print("-" * 60)
    print(document.strip())
    print("-" * 60)
    
    print("\nProcessing flow...")
    
    try:
        # NOTE: We enable use_ministral to show the power of LLM-based detection.
        # we leave gliner False by default as it is an optional dependency.
        result = run_flow(
            document=document,
            confidence_threshold=0.6,
            human_in_loop=False,
            preset='clinical',
            use_ministral=True,  # ENABLE SMART DETECTION
            use_gliner=False,    # OPTIONAL NER (Leave False unless installed)
        )
        
        print("\n" + "=" * 60)
        print("ANONYMIZED OUTPUT:")
        print("=" * 60)
        print(result.get('anonymized_text'))
        print("-" * 60)
        
        # Correctly access detected_entities from FlowState
        entities = result.get('detected_entities', [])
        
        print("\nDETECTION SUMMARY:")
        print(f"Total entities found: {len(entities)}")
        for e in sorted(entities, key=lambda x: x.start):
             print(f"- {e.text} -> [{e.entity_type}] (Conf: {e.score:.2f})")
             
    except Exception as e:
        print(f"\nError running adversarial flow: {e}")
        print("Ensure 'uv sync' was run and the local 'anoner' fork is editable.")

if __name__ == "__main__":
    main()
