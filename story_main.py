import json
import os
from pathlib import Path

from src.story_mas.graph import app
from src.story_mas.schemas import GraphState, WorldBible


def default_bible() -> WorldBible:
    return WorldBible(
        title="장송의 오케스트라",
        canon_docs=[
            "수도 루멘은 음향 마법 공명으로 질서를 유지한다.",
            "길드 ‘콘서트마스터’는 도시의 공명을 감시하고 튜너들을 인도한다.",
            "금지 유물 ‘에코 코어’는 외곽 폐허 지하의 실링 홀에 봉인되어 있다.",
            "침묵구에서는 모든 공명 주문이 약화된다.",
        ],
        glossary={
            "루멘": "수도 이름",
            "콘서트마스터": "공명 감시 길드",
            "에코 코어": "금지 유물",
            "공명": "음향 마법의 파동",
            "포노 스톤": "공명을 저장하는 광석",
            "하모닉 실드": "공명으로 생성하는 방어막",
            "실링 홀": "봉인 의식 장소",
            "침묵구": "공명을 차단하는 구역",
            "변조술": "공명 파형을 바꾸는 기술",
            "리드 소나": "장거리 공명 탐지 의식",
        },
        style_guide={
            "tone": "장중하지만 간결, 중세-스팀펑크 판타지",
            "age": "15",
            "forbidden": ["고어", "외설", "혐오표현", "지나친 잔혹"],
            "speech": "정중체, 과한 은유 자제",
        },
    )


def save_outputs(state):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 상태 접근 헬퍼: dict도, Pydantic 객체도 안전하게 처리
    def get_field(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)

    def to_dict(x):
        try:
            # Pydantic v2
            from pydantic import BaseModel

            if isinstance(x, BaseModel):
                return x.model_dump()
        except Exception:
            pass
        return x  # 이미 dict이면 그대로

    scenario = state["scenario"] if isinstance(state, dict) else get_field(state, "scenario")
    eval_report = state["eval"] if isinstance(state, dict) else get_field(state, "eval")

    # 1) scenario.json 저장
    with open(out_dir / "scenario.json", "w", encoding="utf-8") as f:
        json.dump(to_dict(scenario), f, ensure_ascii=False, indent=2)

    # 2) eval_report.json 저장
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(to_dict(eval_report), f, ensure_ascii=False, indent=2)

    # 3) outline.md 저장(요약)
    outline = get_field(scenario, "outline", {})
    acts = get_field(outline, "acts", [])

    def val(x, key):
        # 씬이 dict일 수도, Pydantic 객체일 수도 있으므로 안전 접근
        return getattr(x, key, x.get(key)) if isinstance(x, dict) else getattr(x, key, None)

    lines = ["# Outline"]
    for i, act in enumerate(acts, 1):
        lines.append(f"\n## Act {i}")
        for s in act:
            sid = val(s, "id")
            summary = val(s, "summary")
            location = val(s, "location")
            chars = val(s, "characters") or []
            lines.append(f"- [{sid}] {summary} @ {location} ({', '.join(chars)})")

    (out_dir / "outline.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    bible = default_bible()
    init = GraphState(bible=bible, instructions={})
    final = app.invoke(init)
    print("\n".join(final["history"]))
    print("score:", final["eval"].score_overall, "| issues:", len(final["eval"].issues))
    save_outputs(final)
