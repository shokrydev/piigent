# Weakness Taxonomy

The analyzer categorizes failures into actionable types:

| Type | Description | Example |
|------|-------------|---------|
| `OVERLAP_CONFLICT` | Higher-score entity replaces correct detection | LOCATION "53168 Bonn" beats DE_POSTAL_CODE "53168" |
| `COVERAGE_GAP` | No recognizer detects this entity type | AGE patterns like "64-jährige" unrecognized |
| `CONTEXT_DEPENDENCY` | Detection requires specific context words | "Müller" only detected with "Herr" prefix |
| `FORMAT_VARIATION` | Entity format differs from expected pattern | KVNR with spaces "A 123 456 780" |
| `ENTITY_CONFUSION` | Wrong entity type assigned | BSNR detected as LANR (same format) |
| `LEAKAGE` | PII survives anonymization | Partial name remains after full name anonymized |
