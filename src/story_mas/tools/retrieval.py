def retrieve_canon(chunks: list[str], query: str, k: int = 5) -> list[str]:
    # 간단 키워드 스코어링(실전에서는 임베딩 검색 권장)
    q = set(query.lower().split())
    scored = [(c, sum(1 for w in c.lower().split() if w in q)) for c in chunks]
    return [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]
