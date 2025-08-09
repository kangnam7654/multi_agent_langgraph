from typing import List, Set

from langgraph.graph import END, StateGraph

from src.story_mas.schemas import (
    DialogueLine,
    EvalIssue,
    EvalReport,
    GraphState,
    PlotOutline,
    Quest,
    ScenarioDoc,
    Scene,
)
from src.story_mas.tools.llm import gen_scenario_json
from src.story_mas.tools.retrieval import retrieve_canon


def supervisor(state: GraphState) -> GraphState:
    # 최초 지시 또는 QA 결과에 따른 보정
    base = {"target_length": {"dialogue_lines": (10, 20)}, "must_use_glossary_min": 2, "min_link_coverage": 0.8}
    if not state.instructions:
        state.instructions = base
    elif state.eval and state.eval.issues:
        # 이슈 유형별 지시 강화
        for issue in state.eval.issues:
            if issue.type == "glossary":
                state.instructions["must_use_glossary_min"] = 3
            if issue.type == "structure":
                state.instructions["min_link_coverage"] = 0.9
    state.history.append("Supervisor: 지시 설정/업데이트")
    return state


def scenario_writer(state: GraphState) -> GraphState:
    # RAG 근거(간단)
    refs = retrieve_canon(state.bible.canon_docs, "도시 루멘 공명 길드 금지 유물 폐허 지하 봉인")
    draft = gen_scenario_json(
        bible={
            "title": state.bible.title,
            "glossary": state.bible.glossary,
            "style_guide": state.bible.style_guide,
            "canon_refs": refs,
        },
        instructions=state.instructions,
    )
    # JSON→모델
    acts = []
    for act in draft["outline"]["acts"]:
        scenes = [Scene(**s) for s in act["scenes"]]
        acts.append(scenes)
    outline = PlotOutline(
        acts=acts,
        themes=draft["outline"].get("themes", []),
        conflicts=draft["outline"].get("conflicts", []),
        payoffs=draft["outline"].get("payoffs", []),
    )
    quests = [Quest(**q) for q in draft["quests"]]
    dialogues = [DialogueLine(**d) for d in draft["dialogues"]]
    state.scenario = ScenarioDoc(outline=outline, quests=quests, dialogues=dialogues)
    state.history.append(f"Writer: 초안 생성(장면 {sum(len(a) for a in acts)}개, 대사 {len(dialogues)}줄)")
    return state


def canon_qa(state: GraphState) -> GraphState:
    issues: List[EvalIssue] = []
    dlg = state.scenario.dialogues
    gl_keys = list(state.bible.glossary.keys())
    # 용어집 사용률
    hits = sum(any(k in d.text for k in gl_keys) for d in dlg)
    hit_rate = hits / max(1, len(dlg))
    if hit_rate < 0.3:
        issues.append(EvalIssue(type="glossary", message="용어집 사용률 낮음", refs=[f"hit_rate={hit_rate:.2f}"]))

    # 금칙/연령
    forbids = set(state.bible.style_guide.get("forbidden", []))
    if any(any(f in d.text for f in forbids) for d in dlg):
        issues.append(EvalIssue(type="style", message="금칙어 발견"))

    if str(state.bible.style_guide.get("age", "15")) in ["12", "15"]:
        risky = ["잔혹", "과도한 폭력"]
        if any(any(r in d.text for r in risky) for d in dlg):
            issues.append(EvalIssue(type="age", message="연령 등급 위반 가능"))

    # 장면-퀘스트 링크 커버리지
    scene_ids: Set[str] = {s.id for act in state.scenario.outline.acts for s in act}
    linked: Set[str] = set()
    for q in state.scenario.quests:
        linked.update(q.related_scenes)
    coverage = len(linked & scene_ids) / max(1, len(scene_ids))
    if coverage < 0.8:
        issues.append(EvalIssue(type="structure", message="장면-퀘스트 링크 부족", refs=[f"coverage={coverage:.2f}"]))

    # 설정 위반(간단): 세계관 핵심 키워드 최소 1개 이상 등장
    canon_keywords = ["루멘", "콘서트마스터", "에코 코어"]
    corpus = " ".join([d.text for d in dlg] + [q.summary for q in state.scenario.quests])
    if not any(k in corpus for k in canon_keywords):
        issues.append(EvalIssue(type="canon", message="설정 핵심 키워드 미반영"))

    score = max(0.0, 1.0 - 0.15 * len(issues))
    state.eval = EvalReport(
        score_overall=score,
        issues=issues,
        metrics={
            "glossary_hit_rate": hit_rate,
            "link_coverage": coverage,
            "canon_violations": 1.0 if any(i.type == "canon" for i in issues) else 0.0,
        },
    )
    state.history.append(f"QA: score={score:.2f}, issues={len(issues)}")
    return state


graph = StateGraph(GraphState)
graph.add_node("Supervisor", supervisor)
graph.add_node("Writer", scenario_writer)
graph.add_node("QA", canon_qa)

graph.set_entry_point("Supervisor")
graph.add_edge("Supervisor", "Writer")
graph.add_edge("Writer", "QA")


def loop_or_end(state: GraphState):
    return "Supervisor" if (state.eval and state.eval.issues) else END


graph.add_conditional_edges("QA", loop_or_end, {"Supervisor": "Supervisor", END: END})
app = graph.compile()
