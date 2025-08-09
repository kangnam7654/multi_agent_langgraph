from pydantic import BaseModel


class WorldBible(BaseModel):
    title: str
    canon_docs: list[str]
    glossary: dict[str, str]
    style_guide: dict[str, object]  # tone, age, forbidden(list), etc.


class Scene(BaseModel):
    id: str
    summary: str
    location: str
    characters: list[str]
    beats: list[str]


class PlotOutline(BaseModel):
    acts: list[list[Scene]]
    themes: list[str] = []
    conflicts: list[str] = []
    payoffs: list[str] = []


class Quest(BaseModel):
    id: str
    name: str
    summary: str
    prerequisites: list[str] = []
    objectives: list[str] = []
    # rewards: dict[str, float] = {}
    rewards: list[str] = []
    difficulty_tag: str = "normal"
    related_scenes: list[str] = []


class DialogueLine(BaseModel):
    scene_id: str
    speaker: str
    text: str
    emotion: str | None = None
    glossary_refs: list[str] = []


class ScenarioDoc(BaseModel):
    outline: PlotOutline
    quests: list[Quest]
    dialogues: list[DialogueLine]


class EvalIssue(BaseModel):
    type: str
    message: str
    refs: list[str] = []


class EvalReport(BaseModel):
    score_overall: float
    issues: list[EvalIssue]
    metrics: dict[str, float]  # glossary_hit_rate, link_coverage, canon_violations 등


class GraphState(BaseModel):
    bible: WorldBible
    instructions: dict[str, object]  # 목표 분량/톤/등급/테마/임계치
    scenario: ScenarioDoc | None = None
    eval: EvalReport | None = None
    history: list[str] = []
    eval: EvalReport | None = None
    history: list[str] = []
