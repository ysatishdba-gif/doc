import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import pandas as pd

from google import genai
from google.cloud import storage
from google.genai import types
from pydantic import BaseModel, Field, field_validator, model_validator

from norm_temporal import TemporalFormulaResolver

# Silence noisy SDK / HTTP client loggers.
for _name in ("google_genai", "httpx", "httpcore", "urllib3", "google"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import logging
logging.getLogger().setLevel(logging.ERROR)


REPRESENTATIVE_TERMS_PROMPT = """
You are a clinical query analyst. Read the EXPANDED clinical query below, understand its overall nature and intent, and distill it down to the canonical named clinical entities it is fundamentally about.

The expanded query may be verbose — it enumerates specifics, spells out abbreviations, and adds clinical context. Do NOT mirror that verbosity. Step back, understand what the query is really asking about, and name the core entities.

A representative term is a canonical, named clinical entity that a typical patient chart has its own section, list, or record for — diagnoses, medications, tests, procedures, devices, allergies, labs, imaging, billing, insurance, visits, etc.

RULES:
- Understand the NATURE of the query first, then name its subject(s). Do not echo or list every detail the expanded query mentions.
- Return the FEWEST terms that capture what the query is about — typically 1, occasionally 2, rarely 3. NEVER more than 3.
- Collapse enumerated specifics back to their parent entity. If the expanded query lists "blood pressure, heart rate, temperature, respiratory rate, oxygen saturation", the entity is "vital signs" — return that, not the five items.
- Each term must name a DIFFERENT entity. Synonyms, abbreviations, and qualifier-wrapped forms of the same entity collapse to ONE canonical short name.
- Pick the SUBJECT, not modifiers. Apply the deletion test: remove the candidate — if the query still makes clinical sense, it was a modifier (drop it); if it collapses, it is the subject (keep it).
- A term is the canonical short noun (e.g. "CT scan", "metformin", "genetic testing"), NOT a descriptive paraphrase or action phrase.
- Temporal expressions, severities, statuses, and action verbs are NEVER representative terms on their own.
- If the query names no clinical entity, return an empty list.

EXAMPLES:
- Expanded: "Electrocardiogram tracings from the most recent cardiac electrical activity recording" -> ["EKG"]
- Expanded: "Vital signs measurement including blood pressure, heart rate, temperature, respiratory rate, oxygen saturation" -> ["vital signs"]
- Expanded: "Genetic testing including molecular diagnostic analysis, hereditary mutation screening, and chromosomal evaluation" -> ["genetic testing"]
- Expanded: "Computed tomography imaging study documenting the presence of an abdominal hernia" -> ["CT scan", "hernia"]

Return ONLY valid JSON (no markdown, no explanation):
{{
  "representative_terms": ["canonical entity 1", "canonical entity 2"]
}}

Expanded query: {expanded_query}
"""

QUERY_EXPANSION_PROMPT = """
You are an expert medical AI assistant specializing in clinical query expansion.
TASK: Expand the user's query into a comprehensive, detailed clinical description.
INSTRUCTIONS:
1. Expand ALL medical abbreviations to full terms (e.g., HTN → Hypertension, DM → Diabetes Mellitus, SOB → Shortness of Breath)
2. Clarify vague medical terms with specific clinical language — always enumerate specifics when the query names a broad category (vitals, labs, imaging, medications, allergies, implants, etc.)
3. Add relevant medical context based on standard clinical practice, even when no abbreviations are present
4. Identify implicit clinical concepts that should be explicit
5. DO NOT add assumptions beyond reasonable clinical interpretation
6. DO NOT include action verbs like "analyze", "review", "check" unless in original query
7. DO NOT hallucinate information not implied by the query
8. Maintain the original query's intent and scope
9. Make sure the Temporal accept is relevant to the Query context

EXAMPLES:
- "Pt with DM" → "Patient with Diabetes Mellitus"
- "Check vitals" → "Vital signs measurement including blood pressure, heart rate, temperature, respiratory rate, oxygen saturation"
- "Family hx of heart disease" → "Cardiovascular disease in family including coronary artery disease, myocardial infarction, heart failure"
- "SOB on exertion" → "Shortness of breath on exertion"
- "Current medicine list" → "Current active medication list including prescription medications, dosages, frequencies, and routes of administration"
- "Do you have an implanted device?" → "Presence of an implanted medical device, including cardiac implants (pacemaker, defibrillator), orthopedic implants, neurostimulators, or other surgical implants"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "expanded_query": "comprehensive expanded clinical description",
  "abbreviations_expanded": ["list of abbreviations that were expanded"]
}}

User Input: {query}
"""

INTENT_EXTRACTION_PROMPT = """
You are a clinical intent extraction engine for medical document retrieval.

====================================================
CORE PRINCIPLES
====================================================

**Extraction Philosophy:**
When requested for clinical information, extract everything needed to fully understand, act upon, or make decisions about that information safely and effectively.

**Guiding Questions:**
1. What is being requested?
2. What contextual information is inseparable from this concept?
3. What would be incomplete or unsafe without?
4. How is this information naturally organized?

**Inseparability Concept:**
Some information types are inherently connected for safety, understanding, or completeness. When extracting one, consider whether the other is contextually necessary.

====================================================
INTENT GENERATION
====================================================

**Analyze the expanded query and identify distinct clinical intents.**

Each intent represents a clinically independent concept that could be documented or understood separately.

Generate as many intents as the expanded query contains. Let the content guide the count.

====================================================
INTENT STRUCTURE
====================================================

For each intent:

1. **intent_title** - What is this about?
2. **description** - What does this represent and why does it matter?
3. **nature** - What is the primary informational purpose? (Format: [Context] / [Purpose])
4. **sub_natures[]** - What are the distinct dimensions of this information?
5. **final_queries[]** - How would this appear in clinical documents?

====================================================
SUB_NATURE DECOMPOSITION
====================================================

**Core Question: "What are the meaningful aspects of this clinical concept?"**

Structure:
{{
  "category_path": "Broad >> Specific >> Detail",
  "atomic_concepts": ["terminal1", "terminal2"]
}}

**CATEGORY_PATH:**
Think of this as organizing information from general to specific. Each level adds meaningful distinction. Use " >> " as the separator.

Consider: "How would I navigate to this information?"

**ATOMIC_CONCEPTS:**
These are the actual data points - the most specific, granular elements at the end of the navigation path.

Consider: "What are the specific pieces of information needed?"

Include all specific details mentioned: exact values, names, dates, measurements, descriptors.

**Self-contained meaning (IMPORTANT):**
Each atomic_concept must name a concept that is correct ON ITS OWN, without relying on the intent_title or category_path for its meaning. The downstream system extracts each atomic_concept as a standalone phrase with NO surrounding context, so any meaning carried only by the intent or the path is LOST.

If the concept's clinical meaning depends on the role its intent gives it, FOLD that role into the atomic_concept itself:
- Allergy intent: emit "food allergy", "latex allergy", "penicillin allergy" — NOT bare "food", "latex", "penicillin".
- Family-history intent: emit "family history of diabetes" — NOT bare "diabetes".
- Contraindication / intolerance intent: emit "aspirin intolerance" — NOT bare "aspirin".

Do NOT add a role when the atomic is ALREADY a specific, self-explaining clinical concept: "anaphylaxis", "angioedema", "hives", "prescription medications", "dosages", "blood pressure" stay exactly as-is. Never attach a role the intent does not actually assign (do not turn "anaphylaxis" into "anaphylaxis allergy"). When in doubt and the bare term already names the right concept, leave it unchanged.

**Key Understanding:**
- category_path = How to get there (the folders)
- atomic_concepts = What's there (the files - be specific)

**Dimension Identification:**
Consider: "What different types of information exist for this concept?"
- Names and identifiers?
- Measurements and quantities?
- Time-related information?
- Location information?
- Characteristics and qualities?
- Relationships and connections?
- Safety-related information?
- People involved?
- Current state or status?
- Surrounding circumstances?

Extract the dimensions that are present and relevant.

**Grouping Logic:**
If multiple pieces of information answer the same type of question, group them in one sub_nature. Build depth in the category_path rather than creating many shallow sub_natures.

====================================================
FINAL_QUERIES: ATOMIC SPECIFICITY
====================================================

**Purpose:**
Generate concise, atomic-level queries that map to precise clinical concepts without generating excessive CUIs.

**Core Insight:**
Long verbose queries generate too many CUIs. Short atomic queries target specific concepts.

Consider the difference:
-  "metformin 500mg twice daily for diabetes management" → generates 10+ CUIs, which is not required
- ✓ "metformin 500mg" → generates 2-3 focused CUIs
- ✓ "twice daily dosing" → generates 1-2 CUIs

**Atomic Query Principle:**
Each query should represent ONE atomic clinical concept or a tight pairing of inseparable concepts.

Think: "What's the smallest meaningful unit?"
- A specific medication + dose
- A specific measurement + value
- A specific condition + severity
- A specific procedure + site

**Generation Approach:**

Extract atomic_concepts directly as queries. Keep them SHORT and PRECISE.

**Guiding Questions:**
- What's the core medical term?
- Is there ONE essential modifier (dose, site, severity)?
- Can this be made shorter while staying meaningful?
- Will this generate a focused set of CUIs?

**Query Characteristics:**
- 2-5 words, meaningful clinical phrase
- Must infer from nature and sub nature context
- A candidate is invalid if it is only a qualifier, modifier, temporal expression, severity descriptor, or status/action word without naming the  entity it applies to
- If a modifier or status word is clinically important, bundle it with the entity it modifies so the candidate still names a concept
- Use standard medical terminology

**Natural Reasoning:**
- Shorter = fewer CUIs = more precise matching
- Atomic = focused = findable
- Multiple short queries > one long query
- Coverage through quantity, not length

**Coverage:**
Generate multiple atomic queries per intent. Each atomic_concept should appear in at least one query, but keep each query short and focused.

====================================================
REASONING FRAMEWORK
====================================================

**Before finalizing, consider:**

On Completeness:
- Have all distinct intents in the expanded query been identified?
- For each intent, have all relevant dimensions been extracted?
- Is there information that's inseparable from what was extracted?

On Specificity:
- Are atomic_concepts as specific as possible?
- Have actual values been included, not just categories?
- Are queries detailed enough to be useful?

On Structure:
- Does each sub_nature represent a different type of information?
- Are atomic_concepts truly the most granular elements?
- Is information properly organized?

On Utility:
- Would someone find what they need with these queries?
- Are the queries realistic for clinical documentation?
- Do the queries cover all the important atomic_concepts?

User Input: {expanded_query}
Timestamp: {timestamp}
"""

CONTEXTUAL_ENVIRONMENT_PROMPT = """You are a clinical documentation intelligence specialist. You are NOT explaining the concept.
Your role is to map each concept's documentation footprint in medical records and infer appropriate retrieval time windows.

You must complete TWO TASKS in one response:

TASK 1 — RETRIEVAL FACETS
TASK 2 — TEMPORAL INFERENCE

Task 1 must be completed first, as its output grounds Task 2.

====================================================
TASK 1: RETRIEVAL FACETS
====================================================

Return exactly {concept_count} entries in concepts_with_context. Return atomic_concept and intent_title exactly as provided.

The atomic_concept is the search target — do NOT restate it in any facet. The five facets describe the clinical context that helps locate it:

- record_types: WHERE information is documented (pathology_report, radiology_report, progress_note, etc.)
ORDERING REQUIREMENT (CRITICAL):
- Return record_types in STRICT descending order of relevance to the atomic_concept
- First item = most definitive source where this concept is primarily documented
- Last item = least direct / more incidental documentation
- Do NOT output random or arbitrary order
- Do NOT include irrelevant record types just to fill space
- Keep list focused (typically 3–6 items)

Relevance definition:
- Highest: where the concept is directly created, measured, or diagnosed
- Medium: where the concept is interpreted, discussed, or managed
- Lowest: where the concept is summarized or passively mentioned

If unsure, follow clinical workflow:
diagnostic source > specialist interpretation > general documentation > summaries
- author_roles: WHO documents the information (pathologist, radiologist, primary_care_physician, etc.)
- longitudinal_scope: WHEN in the clinical lifecycle the information is documented (diagnostic_workup, active_treatment, follow_up, etc.)
- content_signals: WHAT validates the information (icd_diagnosis_code, loinc_lab_code, measurement_value, structured_finding, etc.)
- clinical_settings: IN WHAT SETTING information is documented (inpatient, outpatient, emergency_department, intensive_care_unit, etc.)

For each concept, reason through its complete documentation footprint:
- Where is this concept substantively recorded? By whom? In what setting?
- Who evaluates or acts on it further? What additional documents are generated?
- If severity or complexity increases, who gets involved? Where does documentation shift?
- How is it tracked or summarized over time? Where does longitudinal documentation live?
- Could it appear across different care settings depending on clinical circumstances?

Include only sources where this concept is substantively documented — where a clinician would find meaningful clinical detail, assessment, or action. Exclude sources where the concept is merely listed, copied, or referenced in passing.

RULES:
- Reason about THIS concept inside THIS intent inside THIS query. Do not emit generic filler lists that apply to every concept.
- Facets must be internally coherent: if cardiologist authors it, record_types should be cardiology-relevant.
- Use short snake_case strings. No tooltips. No prose.
- An empty list means "reasoned and concluded this does not apply." record_types and author_roles should rarely be empty.

====================================================
TASK 2: TEMPORAL INFERENCE
====================================================

Task 2 operates at the intent level and may differ from the lifecycle scope in Task 1.

Produce `temporal_by_intent`.

For each intent:
- Include intent_title (unchanged)
- Include candidates (one per candidate)
- Each candidate must include a `temporal_signal` (LIST)

TEMPORAL SIGNAL (facet definition):
- temporal_signal: WHAT TIME WINDOWS apply for retrieval. Each entry is a string. The actual values are derived from the query's wording and the concept's clinical nature — there is NO fixed vocabulary to draw from.

TEMPORAL SIGNAL TYPES:

Four categories. The values you emit come from the query's own language; do not pull from a predefined list.

- RELATIVE: a duration or recency window expressed in the query's words
- ABSOLUTE: the ISO date-range form of a RELATIVE signal — YYYY-MM-DD to YYYY-MM-DD, anchored to the query's reference date
- STATE: a lifecycle qualifier — whether the concept is being asked about as present, ongoing, resolved, or past
- RANK: an ordering cue — when the query asks for a positional pick rather than all instances

Use the query's wording verbatim when it is unambiguous. Normalize only when the query's wording is vague, and derive the normalization from the concept's documentation cadence — not from a fixed lookup.

EMISSION RULES:

Emit ONLY the forms that are anchored in the query or in the concept's nature. Do NOT fabricate a window to "cover all types."

- Emit RELATIVE only when the query expresses a duration or recency window in its wording.
- Emit ABSOLUTE only when RELATIVE is present — it is the ISO form of that span.
- Emit STATE only when (a) the query frames the concept as a state through its wording or tense, or (b) the concept is inherently a state — meaning its clinical documentation has no useful interpretation without a state qualifier.
- Emit RANK only when the query asks for a positional pick (one or more specific positions in an ordered set).

If the query carries no temporal cue and the concept is not inherently a state, return an empty list. An empty temporal_signal is a valid and often correct answer.

INFERENCE PRINCIPLES:

1. The query's wording is the primary anchor. Verb tense, phrasing, framing, and explicit time words are all signals — read them together.

2. When the query expresses a span vaguely, normalize it using the concept's documentation cadence — how often that modality is typically generated, reviewed, or considered current in clinical workflow. Cadence is a property of the concept type, not a fixed table.

3. Do NOT invent a temporal cue the query does not carry. A query with no temporal language and a non-stateful concept yields an empty list — that is the correct answer, not a failure.

4. For concepts whose clinical documentation requires a state qualifier to be meaningful, emit STATE even if the query does not use a state word — but the qualifier you emit must reflect what the query's framing supports.

5. Match concept type to signal type:
   - Concepts that represent an ongoing condition → STATE matching query framing
   - Concepts that represent past events → STATE or RELATIVE matching query framing
   - Diagnostics and measurements → RELATIVE only if the query carries a recency cue; the span follows the modality's cadence
   - Ranked queries → RANK matching the query's ordering language

6. When uncertain, prefer precision over breadth. An empty or narrow list beats a fabricated wider one.


OUTPUT RULES:

- Each temporal_signal entry is ONE of exactly two forms:

  (A) A TIME WINDOW — a duration or recency span ("last 3 months","previous 2 years") or an ISO date range (YYYY-MM-DD to YYYY-MM-DD).
      Emit these EXACTLY as the query expresses them. Do NOT normalize,relabel, or convert a time window into a concept word. Dates and durations stay verbatim.

  (B) A LIFECYCLE QUALIFIER — when the signal is a state/lifecycle concept rather than a window, emit it as a SINGLE canonical clinical concept string that resolves to a UMLS concept (a CUI-alignable term). It must
      be the bare qualifier alone — not a sentence, not a phrase, not wrapped in quantifiers or articles, and never glued to the concept it modifies.

- Form (B) must be UMLS-alignable. Emit the canonical clinical term a terminology lookup would resolve, NOT the query's surface wording:
    * "all historical encounters" -> "historical" (or "past")
    * "currently taking"          -> "current"
    * "current stage"             -> "current"
    * "history of"                -> "past"
  The clinical noun the qualifier modifies NEVER appears in temporal_signal — that noun is the candidate, documented elsewhere. temporal_signal carries
  only the time window or the lifecycle qualifier.

- Do not emit quantifiers ("all", "any", "every"), articles, or full sentences as temporal_signal entries.
- temporal_signal must be a list of plain strings usable directly for UMLS CUI extraction.
- temporal_signal is about WHEN (time window, or position in time: past,  current, recent, onset, ongoing, planned, first, most-recent). It is NOT
  about the disease's STATE, COURSE, OUTCOME, or SEVERITY — words like  remission, exacerbation, recurrence, relapse, progression, successful,  stable describe the condition, not a time, so they belong to the candidate,
  not here.
- Empty list is valid when no meaningful temporal dimension applies.

====================================================
INPUTS
====================================================

Original query: {original_query}
Expanded query: {expanded_query}

Intents with candidates:
{intents_json}

Concepts to document:
{concepts_json}
"""



# LLM EXTRACTION SCHEMAS


class RepresentativeTermsOutput(BaseModel):
 
    representative_terms: List[str] = Field(default_factory=list)


class QueryExpansionOutput(BaseModel):
    expanded_query: str
    abbreviations_expanded: List[str] = Field(default_factory=list)


class SubNature(BaseModel):
    category_path: str
    atomic_concepts: List[str] = Field(default_factory=list)

    @field_validator("category_path")
    @classmethod
    def normalize_separator(cls, v: str) -> str:
        v = str(v) if not isinstance(v, str) else v
        # Accept both ">>" and "/" as separators, normalise to " >> "
        v = re.sub(r"\s*[/>]{1,2}\s*", " >> ", v).strip()
        if ">>" not in v:
            v = f"{v} >> General"
        return re.sub(r"\s*>>\s*", " >> ", v).strip()


class Intent(BaseModel):
    intent_title: str
    description: str = ""
    nature: str = ""
    sub_natures: List[SubNature] = Field(default_factory=list)
    final_queries: List[str] = Field(default_factory=list)


class IntentExtractionResponse(BaseModel):
    total_intents_detected: int = 0
    intents: List[Intent] = Field(default_factory=list)

    @model_validator(mode="after")
    def fix_intent_count(self) -> "IntentExtractionResponse":
        self.total_intents_detected = len(self.intents)
        return self


# Full internal object after both pipeline steps
class IntentExtractionOutput(BaseModel):
    original_query: str
    expanded_query: str
    total_intents_detected: int = 0
    intents: List[Intent] = Field(default_factory=list)


# Contextual environment schemas

class ConceptContext(BaseModel):


    atomic_concept: str
    intent_title: str = ""
    record_types: List[str] = Field(default_factory=list)
    author_roles: List[str] = Field(default_factory=list)
    longitudinal_scope: List[str] = Field(default_factory=list)
    content_signals: List[str] = Field(default_factory=list)
    clinical_settings: List[str] = Field(default_factory=list)


class CandidateTemporal(BaseModel):
    candidate: str
    temporal_signal: List[str] = Field(default_factory=list)

    @field_validator("temporal_signal", mode="before")
    @classmethod
    def strip_type_prefix(cls, v: Any) -> List[str]:
    
        if not isinstance(v, list):
            return v
        prefix_re = re.compile(
            r"^\s*(?:state|status|relative|absolute|rank)\s*[:\-–]\s*",
            re.IGNORECASE,
        )
        cleaned: List[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            s = prefix_re.sub("", item).strip()
            if s:
                cleaned.append(s)
        return cleaned


class IntentTemporal(BaseModel):
    intent_title: str
    candidates: List[CandidateTemporal] = Field(default_factory=list)


class TemporalExtractionOutput(BaseModel):
    intents: List[IntentTemporal] = Field(default_factory=list)


class ContextualEnvironmentOutput(BaseModel):
    concepts_with_context: List[ConceptContext] = Field(default_factory=list)
    temporal_by_intent: List[IntentTemporal] = Field(default_factory=list)



# OUTPUT SCHEMAS


class RetrievalContext(BaseModel):
    record_types: List[str] = Field(default_factory=list)
    author_roles: List[str] = Field(default_factory=list)
    longitudinal_scope: List[str] = Field(default_factory=list)
    content_signals: List[str] = Field(default_factory=list)
    clinical_setting: List[str] = Field(default_factory=list)


class FlatRetrievalSignals(BaseModel):
    record_types: List[str] = Field(default_factory=list)
    author_roles: List[str] = Field(default_factory=list)
    longitudinal_scope: List[str] = Field(default_factory=list)
    temporal_signal: List[str] = Field(default_factory=list)
    temporal_signal_normalized: List[str] = Field(default_factory=list)
    content_signals: List[str] = Field(default_factory=list)
    clinical_setting: List[str] = Field(default_factory=list)
    context_sentences: List[str] = Field(default_factory=list)

    @classmethod
    def from_retrieval_context(cls, ctx: "RetrievalContext") -> "FlatRetrievalSignals":
   
        def _as_list(lst: List[str]) -> List[str]:
            return list(dict.fromkeys(v.strip() for v in (lst or []) if v and v.strip()))

        return cls(
            record_types=_as_list(ctx.record_types),
            author_roles=_as_list(ctx.author_roles),
            longitudinal_scope=_as_list(ctx.longitudinal_scope),
            temporal_signal=[],
            temporal_signal_normalized=[],
            content_signals=_as_list(ctx.content_signals),
            clinical_setting=_as_list(ctx.clinical_setting),
            context_sentences=[],
        )


class IntentSummaryItem(BaseModel):
    intent_title: str
    description: str


class IntentWithoutQueries(BaseModel):
    intent_title: str
    description: str
    nature: str
    sub_natures: List[SubNature]


class FinalCandidateItem(BaseModel):
    candidate_id: str
    intent_title: str
    nature: str
    sub_nature: str
    candidate: str
    retrieval_signals: FlatRetrievalSignals = Field(default_factory=FlatRetrievalSignals)


class IntentWithContext(BaseModel):
    intent_title: str
    description: str
    nature: str
    sub_natures: List[SubNature]
    final_candidates: List[FinalCandidateItem] = Field(default_factory=list)
    retrieval_signals: FlatRetrievalSignals = Field(default_factory=FlatRetrievalSignals)


class Format_IntentExtraction(BaseModel):
    user_query: str
    intents: List[IntentSummaryItem]
    representative_terms: list = Field(default_factory=list)


class Format_FinalQueries(BaseModel):
    user_query: str
    representative_terms: list = Field(default_factory=list)
    total_candidates: int
    final_candidates: List[FinalCandidateItem]


class Format_NatureBreakdown(BaseModel):
    original_query: str
    total_intents_detected: int
    representative_terms: list = Field(default_factory=list)
    intents: List[IntentWithoutQueries]


class Format_FullPipeline(BaseModel):
    original_query: str
    expanded_query: str
    total_intents_detected: int
    representative_terms: list = Field(default_factory=list)
    intents: List[IntentWithContext]


class Format_MinimalPipeline(BaseModel):

    question: str
    expanded_query: Optional[str] = None
    representative_terms: list = Field(default_factory=list)
    total_intents_detected: Optional[int] = None
    intents: Optional[List[IntentWithContext]] = None




NO_TEMPORAL_PLACEHOLDER = "current"


class ExtractionResult:

    def __init__(
        self,
        original_query: str,
        expansion: QueryExpansionOutput,
        intents: IntentExtractionOutput,
        context: Optional[ContextualEnvironmentOutput],
        temporal: Optional[TemporalExtractionOutput],
        processing_time: float,
        representative_terms: Optional[List[str]] = None,
        resolver: Optional["TemporalFormulaResolver"] = None,
    ):
        self.original_query = original_query
        self.expansion = expansion
        self.intents = intents
        self.context = context
        self.temporal = temporal
        self.processing_time = processing_time
        self._resolver = resolver or TemporalFormulaResolver.empty()
      
        self.representative_terms_list: List[str] = list(representative_terms or [])

        # Build O(1) lookup index for concept contexts
        self._concept_index: Optional[Dict[str, ConceptContext]] = None
        if context and context.concepts_with_context:
            self._concept_index = {
                cc.atomic_concept.strip().lower(): cc
                for cc in context.concepts_with_context
            }

    def _build_context_sentences(
        self,
        candidate: str,
        ctx: RetrievalContext,
    ) -> List[str]:
 

        def _join(items: List[str]) -> str:
            parts = [v.strip().replace("_", " ") for v in (items or []) if v and v.strip()]
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            if len(parts) == 2:
                return f"{parts[0]} or {parts[1]}"
            return ", ".join(parts[:-1]) + f", or {parts[-1]}"

        if not ctx or not (ctx.record_types or ctx.author_roles):
            return []

        docs = _join(ctx.record_types)
        auths = _join(ctx.author_roles)
        setts = _join(ctx.clinical_setting)
        sigs = _join(ctx.content_signals)

        parts = [candidate]
        if docs:
            parts.append(f"documented in {docs}")
        if auths:
            parts.append(f"by {auths}")
        if setts:
            parts.append(f"at {setts}")
        if sigs:
            parts.append(f"capturing {sigs}")

        return [", ".join(parts) + "."]

    # helpers

    @staticmethod
    def _aggregate_facets(
        concepts: List[ConceptContext],
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
     
        records, authors, scope, signals, settings = [], [], [], [], []
        for cc in concepts or []:
            records.extend(cc.record_types or [])
            authors.extend(cc.author_roles or [])
            scope.extend(cc.longitudinal_scope or [])
            signals.extend(cc.content_signals or [])
            settings.extend(cc.clinical_settings or [])
        _dd = lambda xs: list(dict.fromkeys(x for x in xs if x))
        return _dd(records), _dd(authors), _dd(scope), _dd(signals), _dd(settings)

    def _get_concept_context(self, atomic_concept: str) -> Optional[ConceptContext]:
    
        if self._concept_index is None:
            return None
        return self._concept_index.get(atomic_concept.strip().lower())

    def _temporal_for_candidate(self, intent_title: str, candidate: str) -> List[str]:

        if not self.temporal or not self.temporal.intents:
            return [NO_TEMPORAL_PLACEHOLDER]
        ikey = intent_title.strip().lower()
        ckey = candidate.strip().lower()
        for it in self.temporal.intents:
            if it.intent_title.strip().lower() != ikey:
                continue
            own: List[str] = []
            intent_union: List[str] = []
            seen: set = set()
            for co in it.candidates or []:
                sig = list(co.temporal_signal or [])
                if co.candidate.strip().lower() == ckey:
                    own = sig
                for s in sig:
                    k = s.strip().lower()
                    if k and k not in seen:
                        intent_union.append(s)
                        seen.add(k)
            result = own if own else intent_union
            return result if result else [NO_TEMPORAL_PLACEHOLDER]
        return [NO_TEMPORAL_PLACEHOLDER]

    def _normalized_temporal(self, temporal_signals: List[str]) -> List[str]:
        """Map generated temporal_signal strings to their normalized formulas
        (e.g. 'ref_point - 3Y') via the resolver. The 'current' placeholder and
        any signal that doesn't match a known concept contribute nothing."""
        real = [s for s in (temporal_signals or []) if s != NO_TEMPORAL_PLACEHOLDER]
        return self._resolver.resolve_many(real)

    def _build_ctx(self, concept_name: str) -> RetrievalContext:
        cc = self._get_concept_context(concept_name)
        if cc is None:
            return RetrievalContext()
        records, authors, scope, signals, settings = self._aggregate_facets([cc])
        return RetrievalContext(
            record_types=records,
            author_roles=authors,
            longitudinal_scope=scope,
            content_signals=signals,
            clinical_setting=settings,
        )

    # format methods

    def _representative_terms(self) -> List[str]:
        
        terms: List[str] = []
        seen: set = set()
        for t in self.representative_terms_list:
            t_clean = str(t).strip()
            if not t_clean:
                continue
            key = t_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(t_clean)

        # Backstop: never blank when there are intents.
        if not terms and self.intents.intents:
            for i in self.intents.intents:
                t_clean = i.intent_title.strip()
                if not t_clean:
                    continue
                key = t_clean.lower()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(t_clean)
                if len(terms) >= 3:  # mirror the prompt's "never more than 3"
                    break

        return terms

    def to_intent_extraction(self) -> Dict[str, Any]:
        items = [
            IntentSummaryItem(intent_title=i.intent_title, description=i.description)
            for i in self.intents.intents
        ]
        result = Format_IntentExtraction(
            user_query=self.original_query,
            intents=items,
            representative_terms=self._representative_terms(),
        )
        return result.model_dump()

    def to_final_queries(self) -> Dict[str, Any]:
        candidates: List[FinalCandidateItem] = []
        seen: set = set()
        idx = 0
        for i in self.intents.intents:
            for sn in i.sub_natures:
                for c in sn.atomic_concepts:
                    key = c.strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    idx += 1
                    rich_ctx = self._build_ctx(c.strip())
                    signals = FlatRetrievalSignals.from_retrieval_context(rich_ctx)
                    signals.temporal_signal = self._temporal_for_candidate(
                        i.intent_title, c.strip()
                    )
                    signals.temporal_signal_normalized = self._normalized_temporal(
                        signals.temporal_signal
                    )
                    signals.context_sentences = self._build_context_sentences(
                        c.strip(), rich_ctx
                    )
                    candidates.append(
                        FinalCandidateItem(
                            candidate_id=f"fc_{idx:03d}",
                            intent_title=i.intent_title,
                            nature=i.nature,
                            sub_nature=sn.category_path,
                            candidate=c.strip(),
                            retrieval_signals=signals,
                        )
                    )
        result = Format_FinalQueries(
            user_query=self.original_query,
            representative_terms=self._representative_terms(),
            total_candidates=len(candidates),
            final_candidates=candidates,
        )
        return result.model_dump()

    def to_nature_breakdown(self) -> Dict[str, Any]:
        intents_no_queries = [
            IntentWithoutQueries(
                intent_title=i.intent_title,
                description=i.description,
                nature=i.nature,
                sub_natures=i.sub_natures,
            )
            for i in self.intents.intents
        ]
        result = Format_NatureBreakdown(
            original_query=self.intents.original_query,
            total_intents_detected=self.intents.total_intents_detected,
            representative_terms=self._representative_terms(),
            intents=intents_no_queries,
        )
        return result.model_dump()

    @staticmethod
    def _merge_signals(candidates: List[FinalCandidateItem]) -> FlatRetrievalSignals:

        def _union(getter) -> List[str]:
            seen_vals: dict = {}
            for fc in candidates:
                for val in getter(fc.retrieval_signals):
                    val = val.strip() if isinstance(val, str) else val
                    if val and val.lower() not in seen_vals:
                        seen_vals[val.lower()] = val
            return list(seen_vals.values())

        temporal = _union(lambda s: s.temporal_signal)
        real_temporal = [t for t in temporal if t != NO_TEMPORAL_PLACEHOLDER]
        if real_temporal:
            temporal = real_temporal
        elif not temporal:
            # Defensive: every candidate fell through with []; keep contract.
            temporal = [NO_TEMPORAL_PLACEHOLDER]

        return FlatRetrievalSignals(
            record_types=_union(lambda s: s.record_types),
            author_roles=_union(lambda s: s.author_roles),
            longitudinal_scope=_union(lambda s: s.longitudinal_scope),
            temporal_signal=temporal,
            temporal_signal_normalized=_union(lambda s: s.temporal_signal_normalized),
            content_signals=_union(lambda s: s.content_signals),
            clinical_setting=_union(lambda s: s.clinical_setting),
            context_sentences=_union(lambda s: s.context_sentences),
        )

    def to_full_pipeline(self) -> Dict[str, Any]:

        seen: set = set()
        idx = 0

        def _make_candidates(intent: Intent) -> List[FinalCandidateItem]:
            nonlocal idx
            items = []
            for sn in intent.sub_natures:
                for c in sn.atomic_concepts:
                    key = c.strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    idx += 1
                    rich_ctx = self._build_ctx(c.strip())
                    signals = FlatRetrievalSignals.from_retrieval_context(rich_ctx)
                    signals.temporal_signal = self._temporal_for_candidate(
                        intent.intent_title, c.strip()
                    )
                    signals.temporal_signal_normalized = self._normalized_temporal(
                        signals.temporal_signal
                    )
                    signals.context_sentences = self._build_context_sentences(
                        c.strip(), rich_ctx
                    )
                    items.append(
                        FinalCandidateItem(
                            candidate_id=f"fc_{idx:03d}",
                            intent_title=intent.intent_title,
                            nature=intent.nature,
                            sub_nature=sn.category_path,
                            candidate=c.strip(),
                            retrieval_signals=signals,
                        )
                    )
            return items

        intent_blocks = []
        for i in self.intents.intents:
            candidates = _make_candidates(i)
            # Intent-level signals = union of its own candidates' signals
            # (including temporal_signal and context_sentences — deduped).
            intent_signals = self._merge_signals(candidates)
            intent_blocks.append(
                IntentWithContext(
                    intent_title=i.intent_title,
                    description=i.description,
                    nature=i.nature,
                    sub_natures=i.sub_natures,
                    final_candidates=candidates,
                    retrieval_signals=intent_signals,
                )
            )

        result = Format_FullPipeline(
            original_query=self.original_query,
            expanded_query=self.expansion.expanded_query,
            total_intents_detected=self.intents.total_intents_detected,
            representative_terms=self._representative_terms(),
            intents=intent_blocks,
        )
        return result.model_dump()



# PIPELINE CLASS


def _collect_temporal_signals(pr: Dict[str, Any]) -> List[str]:
    """Union of temporal_signal across all intents, deduped, order-preserving.
    Lifts the per-intent / per-candidate temporal signals up to the top level
    so the summary's temporal_signals field is never blank for a clinical
    query (every candidate carries at least the 'current' placeholder)."""
    signals, seen = [], set()
    for intent in pr.get("intents") or []:
        rsig = (intent.get("retrieval_signals") or {}).get("temporal_signal") or []
        for s in rsig:
            if isinstance(s, str):
                k = s.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    signals.append(s.strip())
        # Belt-and-suspenders: also walk candidates.
        for fc in intent.get("final_candidates") or []:
            for s in (fc.get("retrieval_signals") or {}).get("temporal_signal") or []:
                if isinstance(s, str):
                    k = s.strip().lower()
                    if k and k not in seen:
                        seen.add(k)
                        signals.append(s.strip())
    return signals


def _collect_record_types(pr: Dict[str, Any]) -> List[str]:
    """Union of record_types across all intents/candidates, deduped,
    order-preserving. Unlike temporal there is no placeholder backstop —
    record_types is legitimately empty when there is no context."""
    records, seen = [], set()
    for intent in pr.get("intents") or []:
        rsig = (intent.get("retrieval_signals") or {}).get("record_types") or []
        for r in rsig:
            if isinstance(r, str):
                k = r.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    records.append(r.strip())
        for fc in intent.get("final_candidates") or []:
            for r in (fc.get("retrieval_signals") or {}).get("record_types") or []:
                if isinstance(r, str):
                    k = r.strip().lower()
                    if k and k not in seen:
                        seen.add(k)
                        records.append(r.strip())
    return records


class ContextualIntentPipeline:

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        temporal_resolver: Optional[TemporalFormulaResolver] = None,
    ):
        self.project = project
        self.location = location
        self.model_name = model
        self._temporal_resolver = temporal_resolver or TemporalFormulaResolver.empty()
       
        self._tls = threading.local()

    @property
    def client(self) -> "genai.Client":
        c = getattr(self._tls, "client", None)
        if c is None:
            c = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
            self._tls.client = c
        return c

    # helpers

    def _extract_json(self, response) -> Any:
        # Prefer .text only when it is actually populated; an empty .text
        # attribute (present but "") would otherwise skip the candidates
        # fallback and try to parse "". Guard every hop into candidates ->
        # content -> parts -> text, because a blocked / empty response can
        # have parts == None, which would raise a confusing
        # "'NoneType' object is not subscriptable" TypeError.
        text = None
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                text = getattr(parts[0], "text", None)
        if not text:
            text = ""

        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        if not text.strip():
            # Empty / blocked response — raise so the caller falls back
            # cleanly instead of hitting a confusing JSONDecodeError on "".
            raise ValueError("empty response text")
        return json.loads(text)

    def _call_model(
        self,
        prompt: str,
        schema: type,
        fallback=None,
        max_tokens: int = 8192,
    ):
        # One retry before falling back — the failures here are intermittent
        # (empty / truncated responses), so a second attempt usually recovers.
        last_exc = None
        for attempt in range(2):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.2,
                    seed=23,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                )
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
                data = self._extract_json(response)
                if isinstance(data, list):
                    fields = schema.model_fields
                    list_field = next(
                        (k for k, v in fields.items() if "List" in str(v.annotation)),
                        None,
                    )
                    data = {list_field: data} if list_field else data
                return schema.model_validate(data)
            except Exception as e:
                last_exc = e

        # Both attempts failed. Surface it so a blank/empty result isn't
        # silently mistaken for a legitimately empty model response.
        if fallback is not None:
            logging.warning(
                "LLM call failed for %s (%s) after retry; using fallback: %s",
                getattr(schema, "__name__", schema),
                type(last_exc).__name__,
                str(last_exc)[:200],
            )
            return fallback
        raise last_exc

    # -- pipeline steps --

    def extract_representative_terms(
        self, expanded_query: str
    ) -> RepresentativeTermsOutput:
        
        return self._call_model(
            REPRESENTATIVE_TERMS_PROMPT.format(expanded_query=expanded_query),
            RepresentativeTermsOutput,
            fallback=RepresentativeTermsOutput(),
            max_tokens=2048,
        )

    def expand_query(self, query: str) -> QueryExpansionOutput:
        """Step 1 — normalize the query (abbreviations + explicit temporals)."""
        return self._call_model(
            QUERY_EXPANSION_PROMPT.format(query=query),
            QueryExpansionOutput,
            fallback=QueryExpansionOutput(expanded_query=query),
            max_tokens=1024,
        )

    def extract_intents(
        self, original_query: str, expanded_query: str,
    ) -> IntentExtractionOutput:
        """Step 2 — extract clinical intents with sub-natures."""
        prompt = INTENT_EXTRACTION_PROMPT.format(
            expanded_query=expanded_query,
            timestamp=datetime.utcnow().isoformat(),
        )
        raw: IntentExtractionResponse = self._call_model(
            prompt,
            IntentExtractionResponse,
            fallback=IntentExtractionResponse(),
            max_tokens=8192,
        )
        return IntentExtractionOutput(
            original_query=original_query,
            expanded_query=expanded_query,
            total_intents_detected=len(raw.intents),
            intents=raw.intents,
        )

    def build_context(
        self,
        query: str,
        expanded_query: str,
        intent_out: IntentExtractionOutput,
    ) -> Optional[ContextualEnvironmentOutput]:
        """Step 3 — map each atomic concept's documentation facets AND
        produce lifecycle qualifiers per intent / candidate in one LLM call."""
        if not intent_out.intents:
            return None

        # Flat list of unique concepts (Task 1 target list)
        concepts: List[Dict[str, str]] = []
        seen: set = set()
        for i in intent_out.intents:
            for sn in i.sub_natures:
                for c in sn.atomic_concepts:
                    k = c.strip().lower()
                    if k and k not in seen:
                        concepts.append(
                            {"atomic_concept": c.strip(), "intent_title": i.intent_title}
                        )
                        seen.add(k)
        if not concepts:
            return None

        # Rich per-intent structure (for Task 2 lifecycle reasoning)
        intents_payload: List[Dict[str, Any]] = []
        for i in intent_out.intents:
            cand_entries: List[str] = []
            seen_c: set = set()
            for sn in i.sub_natures:
                for c in sn.atomic_concepts:
                    k = c.strip().lower()
                    if k and k not in seen_c:
                        cand_entries.append(c.strip())
                        seen_c.add(k)
            intents_payload.append({
                "intent_title": i.intent_title,
                "description": i.description,
                "nature": i.nature,
                "sub_natures": [
                    {"category_path": sn.category_path,
                     "atomic_concepts": list(sn.atomic_concepts or [])}
                    for sn in i.sub_natures
                ],
                "candidates": cand_entries,
            })

        return self._call_model(
            CONTEXTUAL_ENVIRONMENT_PROMPT.format(
                original_query=query,
                expanded_query=expanded_query,
                concepts_json=json.dumps(concepts, indent=2),
                concept_count=len(concepts),
                intents_json=json.dumps(intents_payload, indent=2),
            ),
            ContextualEnvironmentOutput,
            fallback=ContextualEnvironmentOutput(),
            max_tokens=16384,
        )

    # -- main entry points --

    def extract(self, query: str, verbose: bool = False) -> ExtractionResult:
        if not query or not query.strip():
            raise ValueError("Empty query provided")

        t0 = datetime.utcnow()

        # Step 1 — query normalization
        expansion = self.expand_query(query)

        # Step 2 — intent extraction + representative terms. Both depend
        # only on the expanded query, so run them in parallel.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_int = ex.submit(
                self.extract_intents, query, expansion.expanded_query
            )
            f_rep = ex.submit(
                self.extract_representative_terms, expansion.expanded_query
            )
            intent_out = f_int.result()
            rep_terms = f_rep.result()

        # Step 3 — contextual environment (facets + lifecycle qualifier in one call)
        context = self.build_context(query, expansion.expanded_query, intent_out)

        temporal: Optional[TemporalExtractionOutput] = None
        if context is not None:
            temporal = TemporalExtractionOutput(
                intents=list(context.temporal_by_intent or [])
            )

        processing_time = (datetime.utcnow() - t0).total_seconds()

        if verbose:
            candidate_count = sum(
                len(sn.atomic_concepts)
                for i in intent_out.intents
                for sn in i.sub_natures
            )
            sub_nature_count = sum(len(i.sub_natures) for i in intent_out.intents)
            print(f"Query     : {query[:80]}")
            print(f"Expanded  : {expansion.expanded_query[:80]}")
            print(
                f"Intents: {intent_out.total_intents_detected} | "
                f"sub natures: {sub_nature_count} | "
                f"candidates: {candidate_count} | "
                f"time: {processing_time:.1f}s"
            )
            print()

        return ExtractionResult(
            query, expansion, intent_out, context, temporal, processing_time,
            rep_terms.representative_terms, resolver=self._temporal_resolver,
        )

    def run(
        self,
        query: str,
        output_format: Literal[
            "intent_extraction",
            "final_queries",
            "nature_breakdown",
            "full_pipeline",
        ] = "full_pipeline",
    ) -> Dict[str, Any]:
        try:
            result = self.extract(query)
            dispatch = {
                "intent_extraction": result.to_intent_extraction,
                "final_queries": result.to_final_queries,
                "nature_breakdown": result.to_nature_breakdown,
                "full_pipeline": result.to_full_pipeline,
            }
            return dispatch[output_format]()
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
            }




def save_json(data: Dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")


def validate_schema(result: Dict, format_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that an emitted dict matches its declared output schema.
    Returns (ok, error_msg) — error_msg is None on success."""
    schema_map = {
        "intent_extraction": Format_IntentExtraction,
        "final_queries": Format_FinalQueries,
        "nature_breakdown": Format_NatureBreakdown,
        "full_pipeline": Format_FullPipeline,
        "minimal_pipeline": Format_MinimalPipeline,
    }
    schema = schema_map.get(format_type)
    if schema is None:
        return False, f"Unknown format type: {format_type}"
    try:
        schema.model_validate(result)
        return True, None
    except Exception as e:
        return False, str(e)


@dataclass
class OutputConfig:
    intent_extraction: bool = True
    final_queries: bool = True
    nature_breakdown: bool = True
    full_pipeline: bool = True

    def enabled(self) -> List[str]:
        """Return list of format names that are turned on."""
        return [
            name
            for name in (
                "intent_extraction",
                "final_queries",
                "nature_breakdown",
                "full_pipeline",
            )
            if getattr(self, name)
        ]


def run_single_query(
    pipeline: ContextualIntentPipeline,
    query: str,
    idx: int,
    config: Optional[OutputConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = OutputConfig()
    enabled = config.enabled()
    if not enabled:
        print(f"  No outputs enabled for query {idx} — skipping.")
        return {"query": query, "status": "skipped", "files": []}
    result = pipeline.extract(query, verbose=True)
    # Build only the formats that are switched on
    format_dispatch = {
        "intent_extraction": result.to_intent_extraction,
        "final_queries": result.to_final_queries,
        "nature_breakdown": result.to_nature_breakdown,
        "full_pipeline": result.to_full_pipeline,
    }
    prefix = f"query{idx}"
    files: List[str] = []
    for name in enabled:
        data = format_dispatch[name]()
        filename = f"{prefix}_{name}.json"
        save_json(data, filename)
        valid, err = validate_schema(data, name)
        if not valid:
            print(f"  Schema validation failed for {filename}: {err}")
        files.append(filename)
    skipped = [n for n in format_dispatch if n not in enabled]
    if skipped:
        print(f"  Skipped outputs: {', '.join(skipped)}")
    print("-" * 60)
    return {"query": query, "status": "completed", "files": files}


def minimal_pipeline(pipeline, q) -> Dict[str, Any]:
   
    t0 = datetime.utcnow()

    expansion = pipeline.expand_query(q)

    # intents + representative terms both depend only on the expanded query.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_int = ex.submit(pipeline.extract_intents, q, expansion.expanded_query)
        f_rep = ex.submit(
            pipeline.extract_representative_terms, expansion.expanded_query
        )
        intent_out = f_int.result()
        rep_terms = f_rep.result()

    context = pipeline.build_context(q, expansion.expanded_query, intent_out)

    temporal: Optional[TemporalExtractionOutput] = None
    if context is not None:
        temporal = TemporalExtractionOutput(
            intents=list(context.temporal_by_intent or [])
        )

    processing_time = (datetime.utcnow() - t0).total_seconds()
    result = ExtractionResult(
        q, expansion, intent_out, context, temporal, processing_time,
        rep_terms.representative_terms, resolver=pipeline._temporal_resolver,
    )

    fp = result.to_full_pipeline()
    # print(f"Total time: {processing_time:.1f}s")

    return Format_MinimalPipeline(
        question=q,
        expanded_query=fp.get("expanded_query"),
        representative_terms=fp.get("representative_terms"),
        total_intents_detected=fp.get("total_intents_detected"),
        intents=fp.get("intents"),
    ).model_dump()


if __name__ == "__main__":

    PROJECT_ID = PROJECT_ID
    LOCATION = "us-central1"
    MODEL_VERSION = "gemini-2.5-flash"

    # To attach normalized temporal formulas, build a resolver from the table
    # produced by norm_temporal.py and pass it in. Without it, the
    # pipeline behaves exactly as before (temporal_signal_normalized stays []).
    #   from google.cloud import storage
    #   resolver = TemporalFormulaResolver.from_gcs(
    #       storage.Client(project=PROJECT_ID), BUCKET_NAME,
    #       "Normalized_loinc_classes/temporal_norm_code_to_formula.json",
    #   )
    #   # or, from a local copy of the table:
    #   # resolver = TemporalFormulaResolver.from_file("temporal_norm_code_to_formula.json")
    pipeline = ContextualIntentPipeline(
        project=PROJECT_ID, location=LOCATION, model=MODEL_VERSION,
        # temporal_resolver=resolver,
    )


    #  Single query test 

    test_queries = [
"elevated psa","tell me a joke"
    ]
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(minimal_pipeline, pipeline, query): query
            for query in test_queries
        }
        for idx, future in enumerate(as_completed(futures), 2):
            res = future.result()
            results.append(res)
    
            valid, err = validate_schema(res, "minimal_pipeline")
            filename = f"query{idx}_pipeline.json"
            save_json(res, filename)
    
            # Compact highlights — not the full payload.
            print(f"Query     : {res.get('question', '')[:80]}")
            rep = res.get("representative_terms") or []
            print(
                f"Status    : "
                f"intents: {res.get('total_intents_detected')}"
            )
            print(f"Rep terms : {', '.join(rep) if rep else '—'}")
            if not valid:
                print(f"Schema    : FAILED — {err}")
            print("-" * 60)

  
     # From GCS
   

    
#     BUCKET_NAME = BUCKET_NAME
#     FILE_PATH = FILE_PATH
#     BATCH_WORKERS = 4
#     OUTPUT_FILE = "rep-term.json"

#     def load_json_from_gcs(bucket_name, blob_name):
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(blob_name)
#         data = blob.download_as_text()
#         return json.loads(data)

#     def _summarize(pr: Dict[str, Any]) -> Dict[str, Any]:
#         return {
#             "query": pr.get("question", ""),
#             "representative_terms": list(pr.get("representative_terms") or []),
#             "temporal_signals": _collect_temporal_signals(pr),
#             "record_types": _collect_record_types(pr),
#         }

#     data = load_json_from_gcs(BUCKET_NAME, FILE_PATH)
#     questions = list(data.keys())
#     print(f"Total questions: {len(questions)}")

#     if not questions:
#         raise SystemExit(1)

#     summaries: List[Optional[Dict[str, Any]]] = [None] * len(questions)
#     with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
#         future_to_idx = {
#             executor.submit(minimal_pipeline, pipeline, q): i
#             for i, q in enumerate(questions)
#         }
#         for fut in as_completed(future_to_idx):
#             i = future_to_idx[fut]
#             try:
#                 summaries[i] = _summarize(fut.result())
#             except Exception as e:
#                 summaries[i] = {
#                     "query": questions[i],
#                     "representative_terms": [],
#                     "temporal_signals": [],
#                     "record_types": [],
#                     "error": str(e),
#                 }

#     summaries = [s for s in summaries if s is not None]

#     for s in summaries:
#         print(f"Q : {s['query']}")
#         print(f"  rep terms  : {s['representative_terms']}")
#         print(f"  temporal   : {s['temporal_signals']}")
#         print(f"  records    : {s['record_types']}")

#     with open(OUTPUT_FILE, "w") as f:
#         json.dump(summaries, f, indent=2)
#     print(f"Saved {len(summaries)} -> {OUTPUT_FILE}")


     # From csv

#     CSV_FILE_PATH = "shopping_list1.csv"
#     OUTPUT_FILE = "decision-tree-derived.json"
#     BATCH_WORKERS = 16  

#     df = pd.read_csv(CSV_FILE_PATH, encoding="cp1252")
#     questions = df["Shopping List Item"].dropna().astype(str).tolist()
#     print(f"Total questions: {len(questions)}")

#     def _process_one(q: str) -> Dict[str, Any]:
#         try:
#             pr = minimal_pipeline(pipeline, q)
#             return {
#                 "query": q,
#                 "representative_terms": pr.get("representative_terms", []),
#                 "temporal_signals": _collect_temporal_signals(pr),
#                 "record_types": _collect_record_types(pr),
#             }
#         except Exception as e:
#             print(f"Failed: {q} ({e})")
#             return {
#                 "query": q,
#                 "representative_terms": [], "temporal_signals": [],
#                 "record_types": [], "error": str(e),
#             }

#     # Preserve input order in the output
#     results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
#     with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
#         future_to_idx = {
#             executor.submit(_process_one, q): i for i, q in enumerate(questions)
#         }
#         for fut in as_completed(future_to_idx):
#             results[future_to_idx[fut]] = fut.result()

#     results = [r for r in results if r is not None]

#     with open(OUTPUT_FILE, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"Saved {len(results)} -> {OUTPUT_FILE}")
