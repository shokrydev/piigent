"""Section Parser for German Clinical Documents.

Parses clinical documents into semantic sections to enable
section-aware entity detection with appropriate expectations
for each section type.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ClinicalSection(str, Enum):
    """Semantic sections in German clinical documents."""
    HEADER = "header"                   # Document header, metadata
    PATIENT_DATA = "patient_data"       # Patient identification
    ANAMNESIS = "anamnesis"             # Medical history
    SOCIAL_HISTORY = "social_history"   # Social/occupational history
    FAMILY_HISTORY = "family_history"   # Family medical history
    DIAGNOSIS = "diagnosis"             # Diagnosis section
    THERAPY = "therapy"                 # Treatment/therapy
    FINDINGS = "findings"               # Clinical findings
    DISCHARGE = "discharge"             # Discharge summary
    SIGNATURE = "signature"             # Signatures, authorship
    UNKNOWN = "unknown"                 # Unclassified text


# German section headers mapped to section types
SECTION_HEADERS = {
    # Patient data
    r"patient(endaten|in)?:?": ClinicalSection.PATIENT_DATA,
    r"(persönliche\s+)?daten:?": ClinicalSection.PATIENT_DATA,
    r"stammdaten:?": ClinicalSection.PATIENT_DATA,

    # Anamnesis
    r"anamnese:?": ClinicalSection.ANAMNESIS,
    r"eigenanamnese:?": ClinicalSection.ANAMNESIS,
    r"fremdanamnese:?": ClinicalSection.ANAMNESIS,
    r"(aktuelle\s+)?beschwerden:?": ClinicalSection.ANAMNESIS,
    r"vorgeschichte:?": ClinicalSection.ANAMNESIS,

    # Social history
    r"sozialanamnese:?": ClinicalSection.SOCIAL_HISTORY,
    r"soziale\s+anamnese:?": ClinicalSection.SOCIAL_HISTORY,
    r"beruf(liche\s+anamnese)?:?": ClinicalSection.SOCIAL_HISTORY,
    r"lebensumst(ä|ae)nde:?": ClinicalSection.SOCIAL_HISTORY,

    # Family history
    r"familienanamnese:?": ClinicalSection.FAMILY_HISTORY,
    r"famili(ä|ae)re\s+anamnese:?": ClinicalSection.FAMILY_HISTORY,

    # Diagnosis
    r"diagnose(n)?:?": ClinicalSection.DIAGNOSIS,
    r"(haupt|neben)?diagnose:?": ClinicalSection.DIAGNOSIS,
    r"befund:?": ClinicalSection.DIAGNOSIS,

    # Therapy
    r"therapie:?": ClinicalSection.THERAPY,
    r"behandlung:?": ClinicalSection.THERAPY,
    r"medikation:?": ClinicalSection.THERAPY,
    r"massnahmen:?": ClinicalSection.THERAPY,
    r"(operative\s+)?eingriff(e)?:?": ClinicalSection.THERAPY,

    # Findings
    r"befunde?:?": ClinicalSection.FINDINGS,
    r"untersuchungsergebnis(se)?:?": ClinicalSection.FINDINGS,
    r"labor(befunde?|werte)?:?": ClinicalSection.FINDINGS,
    r"(körperliche\s+)?untersuchung:?": ClinicalSection.FINDINGS,

    # Discharge
    r"(ent)?lass(ungs)?(brief|bericht)?:?": ClinicalSection.DISCHARGE,
    r"epikrise:?": ClinicalSection.DISCHARGE,
    r"zusammenfassung:?": ClinicalSection.DISCHARGE,

    # Signature
    r"(behandelnder\s+)?(arzt|ärztin):?": ClinicalSection.SIGNATURE,
    r"unterschrift(en)?:?": ClinicalSection.SIGNATURE,
    r"gez\.": ClinicalSection.SIGNATURE,
    r"mit\s+freundlichen\s+gr(ü|ue)(ß|ss)en": ClinicalSection.SIGNATURE,
}

# Expected entity types for each section
SECTION_ENTITY_EXPECTATIONS = {
    ClinicalSection.HEADER: ["DATE_TIME", "ORGANIZATION"],
    ClinicalSection.PATIENT_DATA: [
        "PERSON", "DE_KVNR", "DATE_TIME", "DE_POSTAL_CODE",
        "AGE", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS"
    ],
    ClinicalSection.ANAMNESIS: ["DATE_TIME", "AGE", "LOCATION", "ORGANIZATION"],
    ClinicalSection.SOCIAL_HISTORY: ["OCCUPATION", "AGE", "LOCATION", "ORGANIZATION"],
    ClinicalSection.FAMILY_HISTORY: ["PERSON", "AGE", "LOCATION"],
    ClinicalSection.DIAGNOSIS: ["DATE_TIME"],
    ClinicalSection.THERAPY: ["DATE_TIME", "ORGANIZATION", "PERSON"],
    ClinicalSection.FINDINGS: ["DATE_TIME"],
    ClinicalSection.DISCHARGE: ["DATE_TIME", "LOCATION", "ORGANIZATION"],
    ClinicalSection.SIGNATURE: ["PERSON", "DE_LANR", "DE_BSNR", "DATE_TIME"],
    ClinicalSection.UNKNOWN: [],  # No special expectations
}


@dataclass
class ParsedSection:
    """A parsed section of a clinical document."""
    section_type: ClinicalSection
    text: str
    start: int  # Character offset in original document
    end: int
    header: Optional[str] = None  # The header text that identified this section
    expected_entities: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set expected entities based on section type."""
        if not self.expected_entities:
            self.expected_entities = SECTION_ENTITY_EXPECTATIONS.get(
                self.section_type, []
            )


@dataclass
class ParsedDocument:
    """A fully parsed clinical document with sections."""
    original_text: str
    sections: List[ParsedSection]

    def get_section_at(self, position: int) -> Optional[ParsedSection]:
        """Get the section containing a given character position."""
        for section in self.sections:
            if section.start <= position < section.end:
                return section
        return None

    def get_sections_by_type(self, section_type: ClinicalSection) -> List[ParsedSection]:
        """Get all sections of a given type."""
        return [s for s in self.sections if s.section_type == section_type]

    def is_entity_expected(self, entity_type: str, position: int) -> bool:
        """Check if an entity type is expected at a given position.

        Args:
            entity_type: The entity type to check
            position: Character position in the document

        Returns:
            True if the entity type is expected in this section
        """
        section = self.get_section_at(position)
        if section is None:
            return True  # Default to allowing if no section

        # If no expectations for this section, allow anything
        if not section.expected_entities:
            return True

        return entity_type in section.expected_entities


class SectionParser:
    """Parser for German clinical documents into semantic sections.

    The parser identifies section boundaries using German medical
    terminology and headers, then classifies each section by type.

    Example:
        parser = SectionParser()
        doc = parser.parse('''
        Patient: Max Mustermann
        KVNR: A123456789

        Sozialanamnese:
        Der Patient arbeitet als Ingenieur.
        ''')

        for section in doc.sections:
            print(f"{section.section_type}: {section.text[:50]}...")
    """

    def __init__(
        self,
        custom_headers: Optional[Dict[str, ClinicalSection]] = None,
    ):
        """Initialize the section parser.

        Args:
            custom_headers: Additional section headers to recognize
        """
        self.headers = dict(SECTION_HEADERS)
        if custom_headers:
            self.headers.update(custom_headers)

        # Compile regex patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), section)
            for pattern, section in self.headers.items()
        ]

    def parse(self, text: str) -> ParsedDocument:
        """Parse a clinical document into sections.

        Args:
            text: The full document text

        Returns:
            ParsedDocument with identified sections
        """
        # Find all section boundaries
        boundaries = self._find_section_boundaries(text)

        # If no sections found, treat entire document as UNKNOWN
        if not boundaries:
            return ParsedDocument(
                original_text=text,
                sections=[ParsedSection(
                    section_type=ClinicalSection.UNKNOWN,
                    text=text,
                    start=0,
                    end=len(text),
                )],
            )

        # Build sections from boundaries
        sections = []
        for i, (start, section_type, header) in enumerate(boundaries):
            # End is either start of next section or end of document
            if i + 1 < len(boundaries):
                end = boundaries[i + 1][0]
            else:
                end = len(text)

            section_text = text[start:end]

            sections.append(ParsedSection(
                section_type=section_type,
                text=section_text,
                start=start,
                end=end,
                header=header,
            ))

        # Handle text before first section
        if boundaries and boundaries[0][0] > 0:
            header_section = ParsedSection(
                section_type=ClinicalSection.HEADER,
                text=text[:boundaries[0][0]],
                start=0,
                end=boundaries[0][0],
            )
            sections.insert(0, header_section)

        return ParsedDocument(
            original_text=text,
            sections=sections,
        )

    def _find_section_boundaries(
        self,
        text: str,
    ) -> List[Tuple[int, ClinicalSection, str]]:
        """Find all section boundaries in text.

        Returns:
            List of (position, section_type, header_text) tuples
        """
        boundaries = []

        # Look for section headers at line beginnings or after double newlines
        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            stripped = line.strip()

            for pattern, section_type in self._compiled_patterns:
                if pattern.match(stripped):
                    boundaries.append((current_pos, section_type, stripped))
                    break

            current_pos += len(line) + 1  # +1 for newline

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        return boundaries

    def get_expected_entities(self, section_type: ClinicalSection) -> List[str]:
        """Get expected entity types for a section.

        Args:
            section_type: The type of section

        Returns:
            List of expected entity type strings
        """
        return SECTION_ENTITY_EXPECTATIONS.get(section_type, [])

    def classify_text_snippet(self, text: str) -> ClinicalSection:
        """Classify a text snippet into a section type.

        This uses heuristics to guess the section type for text
        that doesn't have clear headers.

        Args:
            text: Text snippet to classify

        Returns:
            Most likely section type
        """
        text_lower = text.lower()

        # Look for keywords
        keywords = {
            ClinicalSection.SOCIAL_HISTORY: [
                "beruf", "arbeit", "tätigkeit", "rauchen", "alkohol",
                "wohnt", "lebt", "verheiratet", "ledig"
            ],
            ClinicalSection.FAMILY_HISTORY: [
                "mutter", "vater", "eltern", "großeltern", "geschwister",
                "familiär"
            ],
            ClinicalSection.ANAMNESIS: [
                "beschwerden", "symptom", "schmerz", "seit", "begann"
            ],
            ClinicalSection.DIAGNOSIS: [
                "diagnose", "icd", "befund", "zustand"
            ],
            ClinicalSection.THERAPY: [
                "therapie", "behandlung", "medikament", "op", "operation"
            ],
        }

        scores = {section: 0 for section in ClinicalSection}

        for section, words in keywords.items():
            for word in words:
                if word in text_lower:
                    scores[section] += 1

        # Return section with highest score, or UNKNOWN if all zero
        best_section = max(scores, key=scores.get)
        if scores[best_section] == 0:
            return ClinicalSection.UNKNOWN

        return best_section
