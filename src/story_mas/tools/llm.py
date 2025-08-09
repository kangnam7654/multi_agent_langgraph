import json
import os
from typing import Any, Dict

import requests

USE_LLM = True


def _ollama_client():
    class OllamaClient:
        def chat(self, messages: list[dict[str, str]]):
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": "gpt-oss:20b", "messages": messages, "stream": False},
            )
            response.raise_for_status()

            result = []
            data = {"done": True}
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content")
                    if content:
                        result.append(content)
                if data.get("done"):
                    break
            return "".join(result)

    return OllamaClient()


def gen_scenario_json(bible: Dict[str, Any], instructions: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_LLM:
        # 오프라인 더미(LLM 없으면 간단 초안 반환)
        scene_id = "S000001"
        return {
            "outline": {
                "acts": [
                    [
                        {
                            "id": scene_id,
                            "summary": "도입: 공명 균열 징후",
                            "location": "수도 루멘",
                            "characters": ["주인공", "길드 요원"],
                            "beats": ["징후 포착", "조사 결심"],
                        }
                    ]
                ],
                "themes": ["책임", "균형"],
                "conflicts": ["권력 암투"],
                "payoffs": ["희생의 의미"],
            },
            "quests": [
                {
                    "id": "Q_MAIN_01",
                    "name": "금지된 울림",
                    "summary": "폐허 지하의 ‘에코 코어’ 이상 진동을 조사한다.",
                    "prerequisites": [],
                    "objectives": ["정보 수집", "위치 파악", "퇴로 확보"],
                    "rewards": {"exp": 300, "gold": 100},
                    "difficulty_tag": "normal",
                    "related_scenes": [scene_id],
                }
            ],
            "dialogues": [
                {
                    "scene_id": scene_id,
                    "speaker": "주인공",
                    "text": "루멘의 공명이 흔들려. 누가 ‘에코 코어’를 건드린 거지?",
                },
                {
                    "scene_id": scene_id,
                    "speaker": "길드 요원",
                    "text": "‘콘서트마스터’도 이상 징후를 포착했대. 지금 가자.",
                },
            ],
        }
    # LLM 모드
    sys = (
        "너는 게임 내러티브 작가다. 반드시 주어진 설정집/용어집/스타일가이드만 사용해 사실과 용어를 기술해라. "
        "연령등급과 금칙을 준수하고, 지정한 JSON 스키마만 출력해라."
    )
    prompt = {
        "bible": bible,
        "instructions": instructions,
        "required_schema": {
            "outline": {
                "acts": "3막 권장, 각 막은 Scene 배열",
                "scene_fields": ["id", "summary", "location", "characters", "beats"],
                "themes": "배열",
                "conflicts": "배열",
                "payoffs": "배열",
            },
            "quests": "2~4개, id/name/summary/prerequisites/objectives/rewards/difficulty_tag/related_scenes",
            "dialogues": "10~20줄, scene_id/speaker/text/emotion?/glossary_refs[]",
        },
        "hard_rules": [
            "용어집 키워드 최소 2개 이상 대사에 포함",
            "금칙어 금지, 연령 15 준수",
            "장면과 퀘스트는 서로 링크될 것(related_scenes)",
            "오직 JSON만 출력",
        ],
    }
    client = _ollama_client()
    resp = client.chat(
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
    )
    cleaned = resp.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)
