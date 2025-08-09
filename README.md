# Story MAS (Multi-Agent Scenario Generator)

이 저장소 중 `src/story_mas` 폴더만을 대상으로 한 경량 스토리 시나리오 생성 MAS 설명서입니다.  
LangGraph 기반으로 Supervisor → Writer → QA 루프를 돌며 게임 시나리오(JSON)와 평가 리포트를 산출합니다.

## 핵심 기능
- 세계관 Bible + 용어집 + 스타일가이드 반영
- 시나리오 개요(Acts/Scenes), 퀘스트, 대사(JSON) 자동 생성
- 품질 점검(QA): 용어집 사용률, 링크 커버리지, 금칙/연령, 설정 키워드 반영
- 문제 발견 시 지시 강화를 통한 반복(Feedback Loop)

## 아키텍처 개요
```
GraphState (pydantic)
 ├─ bible (WorldBible)
 ├─ instructions (동적 강화 지시)
 ├─ scenario (ScenarioDoc: outline/quests/dialogues)
 ├─ eval (EvalReport: score/metrics/issues)
 └─ history (노드 수행 로그)

Nodes
 1. Supervisor: 초기/보강 지시 설정
 2. Writer: LLM(or 더미) 호출 → 초안 JSON → 모델 파싱
 3. QA: 규칙 기반 평가 → 이슈 존재 시 Supervisor로 루프
```

## 파일 흐름
| 단계 | 파일 | 역할 |
|------|------|------|
| 실행 | `story_main.py` | 그래프 invoke 및 결과 저장 |
| 그래프 | `src/story_mas/graph.py` | 노드 정의/루프 조건 |
| LLM 호출 | `src/story_mas/tools/llm.py` | Ollama 로컬 Chat → JSON 정제 |
| 평가 | `graph.py::canon_qa` | 규칙 기반 점검 |
| 스키마 | `src/story_mas/schemas.py` | Pydantic 모델 |

## 실행 방법

### 1) 의존 설치 (예시)
```
pip install -e .
```

### 2) Ollama 모델 준비 (LLM 모드)
`tools/llm.py` 기본 설정: 모델 `"gpt-oss:20b"` (예시).  
로컬에서:
```
ollama pull gpt-oss:20b
```

### 3) 실행
```
python story_main.py
```
결과 출력물:
```
outputs/
  ├─ scenario.json      # 전체 시나리오 구조
  ├─ eval_report.json   # 품질 평가 점수/이슈/지표
  └─ outline.md         # Acts/Scenes 요약
```

### 4) 오프라인(더미) 모드
`src/story_mas/tools/llm.py` 상단:
```python
USE_LLM = False
```
로 전환 → 하드코딩된 단일 샘플 구조 반환(빠른 테스트용).

## LLM JSON 안정성 팁
문제: 모델이 코드블록/설명 텍스트 포함하거나 빈 문자열 반환 → `json.loads` 실패.  
현재 처리: 백틱 제거 후 로딩.  
추가 권장:
- 최초 `{` ~ 마지막 `}` 추출 정규식
- 비어있으면 재시도(백오프)
- 실패 시 raw 응답 로그 남김

## 평가 로직(간단 규칙)
- glossary_hit_rate < 0.3 → glossary 이슈
- 금칙어 포함 → style 이슈
- 장면-퀘스트 링크 coverage < 0.8 → structure 이슈
- 핵심 키워드 모두 미포함 → canon 이슈
- 점수 = 1.0 - 0.15 * (#issues) (하한 0)

## 확장 방법

### 새 평가 규칙 추가
`canon_qa` 내 metrics 계산 후:
```python
if <조건>:
    issues.append(EvalIssue(type="custom", message="..."))
```
그리고 `score` 가중치 조정.

### 새 노드 삽입
예: LLM 응답 정규화 노드 추가
1. `graph.add_node("Normalizer", normalizer_fn)`
2. `"Writer" -> "Normalizer" -> "QA"` 엣지 구성

### 지시 강화 로직 변경
`supervisor` 함수 내 이슈 타입별 instruction 조절 로직 편집.

## 주요 스키마 필드 요약
- Scene: id, summary, location, characters, beats
- Quest: related_scenes 로 Scene 연결
- DialogueLine: glossary_refs (모델이 맞춰 넣도록 LLM Prompt 설계)
- EvalReport.metrics: glossary_hit_rate, link_coverage, canon_violations

## 빠른 문제 해결(FAQ)
| 증상 | 원인 | 조치 |
|------|------|------|
| JSONDecodeError | LLM 잡음/빈 응답 | raw 출력 로깅, 정규식 추출 추가 |
| coverage 낮음 | 퀘스트 related_scenes 부족 | Writer 프롬프트에 링크 비율 강조 |
| glossary 사용률 낮음 | LLM 용어 삽입 누락 | Supervisor 가 must_use_glossary_min 증대 |

## 라이선스
MIT (상단 LICENSE 참조)

## 추후 개선 아이디어
- 임베딩 기반 RAG (현재 키워드 매칭)
- 대사 감정 태깅 자동화
- Temperature/Retry 동적 튜닝
- 모델 출력 스트리밍 파서(Partial JSON Recomposer)

---

LinkedIn: [kangnam7654](https://www.linkedin.com/in/kangnam7654)  
Resume: [Link](https://kangnam7654.github.io)
