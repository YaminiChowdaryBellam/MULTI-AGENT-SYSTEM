"""
Input/output guardrail primitives for the clinical research assistant.

- PHI/PII redaction: Presidio, pinned to the small spaCy model (en_core_web_sm)
  rather than Presidio's default en_core_web_lg (~400MB) to keep the footprint
  reasonable for a demo project.
- Prompt-injection detection: rule-based heuristics — deliberately not an LLM
  call, since an injection attempt is exactly the kind of input we don't want
  to hand to an LLM before deciding whether to trust it.
- Out-of-scope check: LLM call — this one needs judgment (is this asking for
  *personalized* treatment advice?) that a keyword list can't reliably make.
"""

import re

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from graph.llm import call_groq, extract_json

# ── PHI / PII redaction ──────────────────────────────────────────────────────
_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}
_nlp_engine = NlpEngineProvider(nlp_configuration=_NLP_CONFIG).create_engine()
_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])
_anonymizer = AnonymizerEngine()

# HIPAA-relevant identifiers Presidio's default recognizers can catch.
PHI_ENTITIES = [
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME",
    "US_SSN", "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "US_PASSPORT",
    "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER", "IP_ADDRESS",
]


def redact_phi(text: str) -> tuple[str, list[dict]]:
    """Returns (redacted_text, redactions) — redactions is [] if nothing was found."""
    results = _analyzer.analyze(text=text, entities=PHI_ENTITIES, language="en")
    if not results:
        return text, []
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    redactions = [
        {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": round(r.score, 2)}
        for r in results
    ]
    return anonymized.text, redactions


# ── Prompt-injection heuristics ──────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all |any |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"reveal (your |the )?(system |initial )?prompt", re.I),
    re.compile(r"pretend (that )?you('re| are)\b", re.I),
    re.compile(r"\bact as (if )?\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bDAN\b"),  # "Do Anything Now" jailbreak persona
    re.compile(r"new instructions:", re.I),
]


def detect_prompt_injection(text: str) -> list[str]:
    """Returns the list of matched pattern strings, [] if none matched."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


# ── Out-of-scope check ───────────────────────────────────────────────────────
def check_out_of_scope(query: str) -> tuple[bool, str]:
    prompt = (
        "You are a scope classifier for a CLINICAL RESEARCH assistant (not a doctor).\n"
        "This assistant answers general medical/clinical research questions (drug info, "
        "literature, trials, guidelines) — it must NOT give personalized treatment advice "
        "for a specific individual's own symptoms or condition.\n\n"
        f"User message: '{query}'\n\n"
        "Is this asking for personalized medical/treatment advice about the user's own "
        "(or a specific named individual's) health situation, symptoms, or what they "
        "personally should do or take? (e.g. 'I have chest pain, what should I do', "
        "'should I stop taking my lisinopril', 'is my child's fever dangerous')\n"
        'Return ONLY JSON: {"out_of_scope": true|false, "reason": "<one sentence>"}'
    )
    result = extract_json(call_groq(prompt)) or {}
    return bool(result.get("out_of_scope", False)), result.get("reason", "")


# ── Output citation enforcement ─────────────────────────────────────────────
_CITATION_TAG_RE = re.compile(r"\[([A-Z]+)\]")
_MIN_CLAIM_WORDS = 8


def _classify_lines(answer: str, known_sources: set[str]) -> list[tuple[str, bool, bool]]:
    """Returns (line, is_substantive_claim, has_valid_citation) per line of the answer."""
    classified = []
    for line in answer.split("\n"):
        text = line.strip()
        if not text:
            classified.append((line, False, False))
            continue
        is_heading = text.startswith("#") or (text.startswith("**") and text.endswith("**"))
        tags = set(_CITATION_TAG_RE.findall(text))
        has_valid_citation = bool(tags & known_sources)
        is_substantive_claim = len(text.split()) >= _MIN_CLAIM_WORDS and not is_heading
        classified.append((line, is_substantive_claim, has_valid_citation))
    return classified


def enforce_citations(answer: str, known_sources: set[str]) -> tuple[str, dict]:
    """
    Strips substantive lines that make claims but carry no citation tag
    pointing to a known specialist source. Short lines, blank lines, and
    markdown headings/bold labels pass through untouched.
    """
    kept_lines = []
    stripped = 0
    for line, is_claim, has_citation in _classify_lines(answer, known_sources):
        if is_claim and not has_citation:
            stripped += 1
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines), {"stripped_claims": stripped}


def citation_coverage(answer: str, known_sources: set[str]) -> tuple[float, dict]:
    """
    Measures (without modifying) what fraction of substantive claim-lines
    already carry a valid citation — used by the eval harness to score how
    well synthesis self-complies before output_guard ever has to intervene.
    Returns 1.0 if the answer has no substantive claims to begin with.
    """
    claims = [(is_claim, has_citation) for _, is_claim, has_citation in _classify_lines(answer, known_sources) if is_claim]
    if not claims:
        return 1.0, {"total_claims": 0, "cited_claims": 0}
    cited = sum(1 for _, has_citation in claims if has_citation)
    return round(cited / len(claims), 4), {"total_claims": len(claims), "cited_claims": cited}


DISCLAIMER = (
    "\n\n---\n*This information is for general clinical research purposes only and is not "
    "personalized medical advice. Always consult a qualified healthcare provider for diagnosis "
    "or treatment decisions.*"
)
