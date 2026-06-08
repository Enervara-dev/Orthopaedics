"""
Offline end-to-end smoke test for the GraphRAG pulmonology assistant.

Runs WITHOUT network or paid calls: Pinecone, Neo4j, and Gemini are stubbed.
It exercises every subsystem and asserts behavior:

  - imports / wiring          (all packages import)
  - domain layer              (namespace, scope threshold, prompts, red flags)
  - session memory            (extraction, symptom-weighted risk, triggers, continuity)
  - entity processor          (plain-name parse, dedup, hybrid merge)
  - full pipeline scenarios   (in-scope, out-of-scope, emergency, terminal state,
                               NO_RETRIEVAL conclude, greeting)
  - Stage-4 prompt injection  (real gemini_llm, generate_stream patched)
  - HTTP API wiring           (best-effort: routes registered)

A LIVE run (real Pinecone/Neo4j/Gemini) still has to be done on a networked
machine — this validates the wiring and decision logic, not the data stores.

Usage:
    python smoke_test.py          # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.ERROR)
# Silence the pipeline's own loggers (their handlers bind to the real stdout and
# bypass redirect_stdout) so the smoke-test report stays readable.
logging.disable(logging.CRITICAL)


# ── tiny harness ──────────────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        ok = bool(cond)
        self.passed += ok
        self.failed += (not ok)
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  → {detail}"
        print(line)
        return ok

    def skip(self, name: str, reason: str) -> None:
        print(f"  [SKIP] {name}  → {reason}")


R = Report()


@contextmanager
def silent():
    """Swallow the pipeline's stdout (stage logs / streamed prints) for a clean report."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield


# ── stub pipeline builder ─────────────────────────────────────────────────────
def build_stub_pipeline(analyses, *, graph=("asthma -[manifests_as]-> wheeze",),
                        chunk_entities=("asthma", "wheeze"), llm_answer="ANSWER: clinical guidance"):
    """
    A GraphRAGPipeline with external services stubbed but real in-process
    components (entity_processor, memory_adapter → RAM). `analyses` is a list of
    gatekeeper-analysis dicts returned in order, one per .run() call.
    Returns (pipeline, calls) where `calls` records each generate_response kwargs.
    """
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.memory import SessionMemoryAdapter

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p._episodic = None
    p._loop = None
    p.entity_processor = EntityProcessor()
    p.memory_adapter = SessionMemoryAdapter()
    calls: list[dict] = []
    flags = {"retrieved": False}

    class A:
        def __init__(self): self.q = list(analyses)
        def analyze(self, q): return self.q.pop(0) if self.q else {}

    class PC:
        def retrieve(self, *a, **k):
            flags["retrieved"] = True
            return [{"id": "c1", "metadata": {
                "summary": "Asthma is an airway disease that causes wheeze.",
                "entities": list(chunk_entities)}}]

    class N:
        def retrieve_relations(self, *a, **k): return list(graph)
        def close(self): pass

    class L:
        def generate_response(self, **k):
            calls.append(k)
            return llm_answer

    p.query_analyzer = A()
    p.pinecone_retriever = PC()
    p.neo4j_retriever = N()
    p.llm = L()
    return p, calls, flags


def analysis(intent="symptom_query", *, needs_followup=False, relevance=95,
             risk="low", action="retrieve", symptoms=("cough",)):
    return {
        "domain": "health", "intent": intent, "risk_level": risk,
        "pulmonology_relevance": relevance,
        "medical_entities": {"symptoms": list(symptoms), "drugs": [], "conditions": []},
        "rewritten_query": "", "needs_followup": needs_followup,
        "followup_questions": (["q"] if needs_followup else []), "final_action": action,
    }


# ── 1. imports / wiring ───────────────────────────────────────────────────────
def test_imports():
    R.section("1. Imports / wiring")
    import importlib
    mods = [
        "graphrag.pipeline.graphrag_pipeline",
        "graphrag.query_understanding.analyzer",
        "graphrag.query_understanding.routing",
        "graphrag.retrievers.pinecone_retriever",
        "graphrag.retrievers.neo4j_retriever",
        "graphrag.processors.entity_processor",
        "graphrag.llm.gemini_llm",
        "graphrag.domain",
        "Memory_Layer.session_memory",
        "Memory_Layer.session_memory.domain",
        "episodic.api.dependencies",
        "episodic.schemas.retrieval",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            R.check(f"import {m}", True)
        except Exception as e:
            R.check(f"import {m}", False, repr(e))


# ── 2. domain layer ───────────────────────────────────────────────────────────
def test_domain():
    R.section("2. Domain layer")
    from graphrag.domain import (
        PINECONE_NAMESPACE, PULMONOLOGY_RELEVANCE_THRESHOLD,
        GATEKEEPER_SYSTEM_PROMPT, compose_system_prompt, detect_red_flags,
    )
    from graphrag.domain.clinical_policy import (
        closure_directive, ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION,
        MAX_DIAGNOSTIC_TURNS,
    )
    R.check("retrieval namespace = pulmonology_v1", PINECONE_NAMESPACE == "pulmonology_v1", PINECONE_NAMESPACE)
    R.check("scope threshold = 75", PULMONOLOGY_RELEVANCE_THRESHOLD == 75)
    R.check("max diagnostic turns = 2", MAX_DIAGNOSTIC_TURNS == 2)
    R.check("gatekeeper prompt has relevance rubric + red flags + terminal state",
            all(s in GATEKEEPER_SYSTEM_PROMPT for s in
                ("pulmonology_relevance", "RESPIRATORY / CARDIOPULMONARY RED FLAGS", "assessment_ready")))
    crit = compose_system_prompt(query_type="symptom_query", risk_level="critical", has_name=False)
    R.check("critical answer prompt = structured emergency",
            "EMERGENCY RESPONSE STRUCTURE" in crit and "POSSIBLE SERIOUS CAUSES" in crit)
    pulm = compose_system_prompt(query_type="symptom_query", risk_level="none", has_name=False)
    R.check("answer prompt is pulmonology-tuned", "pulmonology" in pulm.lower())
    # red flag detection
    R.check("red flag: coughing up blood", detect_red_flags("I am coughing up blood") == ["haemoptysis"])
    R.check("red flag: NOT tripped by mild 'cant breathe properly'",
            detect_red_flags("i cant breath properly, sinus") == [])
    # closure directive matrix
    R.check("closure: greeting (no findings) → none",
            closure_directive(intent="greeting", needs_followup=False, memory_only=True, has_findings=False) is None)
    R.check("closure: assessment_ready → terminal instruction",
            closure_directive(intent="assessment_ready", needs_followup=False, memory_only=False, has_findings=True) == ASSESSMENT_READY_INSTRUCTION)
    R.check("closure: NO_RETRIEVAL medical → conclude instruction",
            closure_directive(intent="followup_query", needs_followup=True, memory_only=True, has_findings=True) == NO_RETRIEVAL_CONCLUDE_INSTRUCTION)
    R.check("closure: mid-triage (needs_followup) → none",
            closure_directive(intent="symptom_query", needs_followup=True, memory_only=False, has_findings=True) is None)


# ── 3. session memory ─────────────────────────────────────────────────────────
def test_memory():
    R.section("3. Session memory")
    from Memory_Layer.session_memory import SessionMemory, Message, Role, extract_state, get_working_memory
    from Memory_Layer.session_memory.state_extractor import extract_entities

    raw = extract_entities("chest pain and wheezing, worse in the morning, coughing up blood")
    R.check("respiratory symptom extraction", {"chest_pain", "wheezing", "haemoptysis"} <= set(raw.symptoms), str(raw.symptoms))
    R.check("trigger extraction", "morning" in raw.triggers, str(raw.triggers))

    # symptom-weighted risk in the live path
    s = SessionMemory(session_id="m1")
    s.state = extract_state(s, Message(role=Role.USER, content="I have chest pain", risk_level="low"))
    R.check("critical symptom escalates risk → critical", str(s.state.risk_level) in ("critical", "RiskLevel.CRITICAL"), str(s.state.risk_level))

    # continuity across turns
    s2 = SessionMemory(session_id="m2")
    s2.state = extract_state(s2, Message(role=Role.USER, content="cough for 3 days"))
    s2.add_turn(Message(role=Role.USER, content="cough for 3 days"))
    s2.state = extract_state(s2, Message(role=Role.USER, content="now also wheezing"))
    R.check("symptoms accumulate across turns", {"cough", "wheezing"} <= set(s2.state.symptoms), str(s2.state.symptoms))


# ── 4. entity processor ───────────────────────────────────────────────────────
def test_entities():
    R.section("4. Entity processor")
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.pipeline.graphrag_pipeline import _merge_graph_entities, _entities_from_analysis

    # plain-name metadata (real Pinecone format) — the bug we fixed
    _, ents, _ = EntityProcessor.process_matches(
        [{"id": "1", "metadata": {"summary": "s", "entities": ["asthma", "wheeze", "asthma"]}}],
        priority_entity_types=["disease"], query="")
    R.check("plain-name entities extracted + deduped", ents == ["asthma", "wheeze"], str(ents))

    merged = _merge_graph_entities(["asthma"], _entities_from_analysis(
        {"medical_entities": {"symptoms": ["chest_pain"], "drugs": [], "conditions": ["copd"]}}), ["wheeze"])
    R.check("hybrid graph entities (chunk+query+memory, normalized)",
            merged[0] == "asthma" and "chest pain" in merged, str(merged))


# ── 5. full pipeline scenarios ────────────────────────────────────────────────
def test_pipeline_scenarios():
    R.section("5. Full pipeline (stubbed services)")
    from graphrag.domain import OUT_OF_SCOPE_MESSAGE

    # a) in-scope → retrieval + graph + answer, graph entities are hybrid
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=True)])
    with silent():
        ans = p.run("breathless and wheezing", session_id="sc_in")
    gctx = calls[-1]["graph_context"]
    R.check("in-scope answered + retrieval ran", ans.startswith("ANSWER:") and flags["retrieved"])
    R.check("graph traversal produced relations", "manifests_as" in gctx, gctx[:60])

    # b) out-of-scope → restricted, retrieval skipped
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", relevance=20, needs_followup=False)])
    with silent():
        ans = p.run("itchy skin rash on my arm", session_id="sc_oos")
    R.check("out-of-scope restricted", ans == OUT_OF_SCOPE_MESSAGE)
    R.check("out-of-scope skipped retrieval", flags["retrieved"] is False and not calls)

    # c) emergency (red flag) → reasoned answer at critical risk, retrieval ran
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=False)])
    with silent():
        ans = p.run("I am coughing up blood and struggling to breathe", session_id="sc_er")
    R.check("emergency → reasoned LLM answer (not static)", ans.startswith("ANSWER:"))
    R.check("emergency → critical risk + retrieval ran",
            calls and calls[-1]["risk_level"] == "critical" and flags["retrieved"], str(calls[-1]["risk_level"]) if calls else "no-call")

    # d) terminal state: 3 follow-needed turns → 3rd flips to assessment_ready
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True)] * 3)
    with silent():
        for _ in range(3):
            p.run("I have a cough", session_id="sc_turns")
    R.check("turn 1 not terminal", calls[0]["query_type"] == "symptom_query")
    R.check("turn 3 forced → assessment_ready", calls[2]["query_type"] == "assessment_ready" and calls[2]["needs_followup"] is False)

    # e) needs_followup False mid-loop → terminal
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("symptom_query", needs_followup=False)])
    with silent():
        p.run("I have a cough", session_id="sc_nf")
        p.run("still coughing", session_id="sc_nf")
    R.check("needs_followup False → assessment_ready", calls[1]["query_type"] == "assessment_ready")

    # f) NO_RETRIEVAL medical follow-up → memory_only + findings (conclude)
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("followup_query", needs_followup=True, action="route_to_followup")])
    with silent():
        p.run("I have a cough", session_id="sc_nr")
        p.run("is it serious?", session_id="sc_nr")
    R.check("NO_RETRIEVAL follow-up → memory_only + has_findings",
            calls[1]["memory_only"] is True and calls[1]["has_findings"] is True, str({k: calls[1][k] for k in ("memory_only", "has_findings")}))

    # g) greeting → exempt from scope gate, answered
    p, calls, _ = build_stub_pipeline([analysis("greeting", relevance=5, needs_followup=False)])
    with silent():
        ans = p.run("hello", session_id="sc_hi")
    R.check("greeting exempt → answered", ans.startswith("ANSWER:"))


# ── 6. Stage-4 prompt injection (real gemini_llm, generate_stream patched) ─────
def test_stage4_injection():
    R.section("6. Stage-4 prompt injection (real gemini_llm)")
    try:
        import graphrag.llm.gemini_llm as gl
        from graphrag.domain.clinical_policy import ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    except Exception as e:
        R.skip("gemini_llm injection", repr(e))
        return

    cap: dict = {}

    def fake_stream(*, user_prompt, model, system_instruction=None, temperature=None):
        cap["sys"] = system_instruction
        yield "ok"

    gl.generate_stream = fake_stream
    try:
        llm = gl.GeminiLLM()
    except Exception as e:
        R.skip("gemini_llm injection (needs GEMINI_API_KEY for client init)", repr(e))
        return

    def call(**kw):
        with silent():
            llm.generate_response(query_text="q", vector_context="", graph_context="",
                                  memory_context="", conversation_history="", **kw)
        return cap.get("sys", "")

    s = call(query_type="assessment_ready", needs_followup=False, memory_only=False, has_findings=True)
    R.check("assessment_ready → terminal constraint injected", ASSESSMENT_READY_INSTRUCTION in s)
    s = call(query_type="followup_query", needs_followup=True, memory_only=True, has_findings=True)
    R.check("NO_RETRIEVAL → conclude constraint injected", NO_RETRIEVAL_CONCLUDE_INSTRUCTION in s)
    s = call(query_type="symptom_query", needs_followup=True, memory_only=False, has_findings=True)
    R.check("mid-triage → NO constraint (follow-up allowed)",
            ASSESSMENT_READY_INSTRUCTION not in s and NO_RETRIEVAL_CONCLUDE_INSTRUCTION not in s)


# ── 7. HTTP API wiring (best-effort) ──────────────────────────────────────────
def test_api_wiring():
    R.section("7. HTTP API wiring")
    try:
        import fastapi  # noqa: F401
    except Exception:
        R.skip("FastAPI app", "fastapi not installed in this interpreter")
        return
    try:
        from app.main import app
        paths = {getattr(r, "path", None) for r in app.routes}
        R.check("/health route registered", "/health" in paths)
        R.check("/chat route registered", "/chat" in paths)
    except Exception as e:
        R.check("app.main imports", False, repr(e))


def main() -> None:
    print("=" * 64)
    print("  OFFLINE SMOKE TEST — GraphRAG pulmonology assistant")
    print("  (Pinecone / Neo4j / Gemini stubbed — no network, no cost)")
    print("=" * 64)
    test_imports()
    test_domain()
    test_memory()
    test_entities()
    test_pipeline_scenarios()
    test_stage4_injection()
    test_api_wiring()
    print("\n" + "=" * 64)
    total = R.passed + R.failed
    print(f"  RESULT: {R.passed}/{total} checks passed, {R.failed} failed")
    print("=" * 64)
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
