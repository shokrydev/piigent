# Key Research Findings

From running PIIgent's weakness exploration on synthetic German clinical documents, the following patterns emerged:

1. **Overlap conflicts are the primary failure mode**
   When PLZ appears adjacent to a city name ("10115 Berlin"), the LLM labels the combined span as LOCATION, overriding the pattern recognizer's more specific DE_POSTAL_CODE detection. Max-score aggregation favors the LLM's higher confidence.

2. **Pattern recognizers are highly reliable when not overridden**
   KVNR, LANR, BSNR achieve ~99% F2 thanks to checksum validation. The failures occur at the aggregation stage, not detection.

3. **Context dependency causes false negatives**
   Standalone surnames ("Müller") are often missed; with honorifics ("Herr Müller") they're detected. The LLM needs contextual cues for German names.

4. **Coverage gaps exist**
   AGE patterns ("64-jährige") have no dedicated recognizer. This is a gap in the recognizer ensemble, not an aggregation conflict.

These findings suggest targeted fixes: implement PREFER_SPECIFIC aggregation for structured IDs, add context boosting for names, create AGE pattern recognizer.
