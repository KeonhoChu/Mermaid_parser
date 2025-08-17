# vLLM Chat Completions 배치 러너 (OpenAI 호환)
# - 엔드포인트: http://10.240.1.8:8001/v1/chat/completions
# - 모델: openai/gpt-oss-120b
# - Mermaid .mmd 파일을 읽어 모델 JSON 응답을 저장하고 CSV로 요약
# - 외부 의존성: requests (pip install requests)

import os
import json
import time
import glob
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests

VLLM_ENDPOINT = "http://10.240.1.8:8001/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"

# -------- 프롬프트 (정규화) --------
SYSTEM_PROMPT = """You are a careful diagram interpreter.
Given a Mermaid flowchart, extract its structure and explain it.
Output strictly in the provided JSON schema. No extra text.
Rules:
- Do NOT invent nodes/edges not present in input.
- Expand '&' fan-out/fan-in into all pairwise edges.
- Keep node IDs exactly as in input; map shapes: () circle, (()) double_circle, [] rect, [[]] double_rect, {} diamond, >> >> subroutine.
- Labeled edges are text between -- and -->.
- Deduplicate and sort outputs as instructed.
"""

USER_TEMPLATE = """[MERMAID_FLOWCHART_START]
{mermaid_code}
[MERMAID_FLOWCHART_END]

Return JSON with this schema ONLY (no prose outside JSON):

{{
  "ok": boolean,
  "warnings": string[],
  "nodes": [
    {{
      "id": string,
      "label": string|null,
      "shape": "rect"|"circle"|"double_rect"|"double_circle"|"diamond"|"subroutine"|null,
      "subgraphs": string[]
    }}
  ],
  "edges": [
    {{
      "source": string,
      "target": string,
      "type": "-->"|"-.-\\u003e"|"==>"|"~~~>",
      "label": string|null
    }}
  ],
  "subgraphs": [{{"id": string, "title": string|null}}],
  "decisions": string[],
  "summary": {{
    "natural_language": string,
    "entry_nodes": string[],
    "exit_nodes": string[]
  }}
}}

Normalization you MUST apply:
- Expand '&' groups (A --> B & C, A & B --> C & D).
- Remove duplicates; sort nodes by id; sort edges by (source,target,type,label).
- Empty strings → null. If unsupported/ambiguous: ok=false and add warnings.
"""

# -------- 유틸: 호출/파싱 --------
def call_vllm(messages: List[Dict[str, str]],
              temperature: float = 0.0,
              top_p: float = 1.0,
              max_tokens: int = 2000,
              seed: int = 123,
              timeout: int = 120,
              retries: int = 3,
              retry_backoff: float = 1.8) -> str:
    """
    vLLM OpenAI 호환 /v1/chat/completions 호출.
    반환: 모델의 text(assistant content)
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed
    }
    last_err = None
    for attempt in range(1, retries+1):
        try:
            resp = requests.post(VLLM_ENDPOINT, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                time.sleep(retry_backoff ** attempt)
                continue
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff ** attempt)
    raise last_err

def strip_code_fences(s: str) -> str:
    """모델이 ```json ... ``` 같은 fence를 둘러줄 경우 제거"""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_llm_json(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    모델 출력에서 JSON을 파싱. 성공 여부, JSON 객체, 경고 문자열을 반환.
    """
    raw = strip_code_fences(text)
    # 흔한 HTML escape 복원
    raw = raw.replace("\\u003e", ">")
    try:
        obj = json.loads(raw)
        return True, obj, ""
    except Exception as e:
        return False, {}, f"json_parse_error: {e}"

def count_nodes_edges(obj: Dict[str, Any]) -> Tuple[int, int]:
    n_nodes = 0
    n_edges = 0
    try:
        if isinstance(obj.get("nodes"), list):
            n_nodes = len(obj["nodes"])
        if isinstance(obj.get("edges"), list):
            n_edges = len(obj["edges"])
    except Exception:
        pass
    return n_nodes, n_edges

# -------- 배치 실행 --------
def run_batch(mermaid_dir: str = "mermaid",
              out_dir: str = "results",
              level_csv: Optional[str] = "mermaid_analysis_results.csv",
              model_name: str = MODEL_NAME):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_dir = Path(out_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # 기존 난이도 CSV(선택) 로드 → name 기준 병합용
    level_map: Dict[str, Dict[str, Any]] = {}
    if level_csv and Path(level_csv).exists():
        import pandas as pd
        df = pd.read_csv(level_csv)
        for _, r in df.iterrows():
            level_map[str(r["name"])] = {
                "level": r.get("level", None),
                "score_final": r.get("score_final", None)
            }

    files = sorted(glob.glob(os.path.join(mermaid_dir, "*.mmd")))
    rows = []

    for fp in files:
        name = Path(fp).stem
        with open(fp, "r", encoding="utf-8") as f:
            code = f.read()

        user_msg = USER_TEMPLATE.format(mermaid_code=code)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]

        try:
            text = call_vllm(messages)
            ok, obj, warn = parse_llm_json(text)
        except Exception as e:
            ok, obj, warn = False, {}, f"request_error: {e}"

        # 요약 통계
        n_nodes_pred, n_edges_pred = (0, 0)
        warnings_str = ""
        if ok:
            n_nodes_pred, n_edges_pred = count_nodes_edges(obj)
            # 모델이 warnings 배열을 제공하면 문자열로 붙이기
            if isinstance(obj.get("warnings"), list):
                warnings_str = ";".join(map(str, obj["warnings"]))
        else:
            warnings_str = warn

        # JSON 저장
        json_path = json_dir / f"{name}_{model_name.replace('/','_')}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                if ok:
                    json.dump(obj, jf, ensure_ascii=False, indent=2)
                else:
                    json.dump({"raw": text if "text" in locals() else "", "error": warnings_str}, jf, ensure_ascii=False, indent=2)
        except Exception:
            # 저장 실패는 무시하고 진행
            pass

        # 레벨/점수 병합 (없으면 공란)
        lv = level_map.get(name, {}).get("level", "")
        sc = level_map.get(name, {}).get("score_final", "")

        rows.append({
            "name": name,
            "model": model_name,
            "level": lv,
            "score_final": sc,
            "llm_ok": ok,
            "llm_warnings": warnings_str,
            "n_nodes_pred": n_nodes_pred,
            "n_edges_pred": n_edges_pred,
            "json_raw_path": str(json_path)
        })

    # CSV 저장
    out_csv = Path(out_dir) / "llm_runs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            "name","model","level","score_final",
            "llm_ok","llm_warnings","n_nodes_pred","n_edges_pred","json_raw_path"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[완료] LLM 결과 CSV 저장: {out_csv}")


if __name__ == "__main__":
    run_batch(
        mermaid_dir="mermaid",
        out_dir="results",
        level_csv="mermaid_analysis_results.csv",  # 없으면 None 또는 파일 삭제
        model_name=MODEL_NAME
    )
