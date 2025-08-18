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
import random

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
    - 200/4xx 모두 JSON 파싱 시도하여 에러 메시지 보존
    - choices/message/delta 폴백 처리
    - 지수+지터 백오프 재시도
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # 일부 서버에서 seed 미지원; 지원 안 하면 서버가 무시하거나 4xx를 낼 수 있음
        "seed": seed,
    }

    headers = {"Content-Type": "application/json"}
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(VLLM_ENDPOINT, json=payload, timeout=timeout, headers=headers)
            # 본문을 최대한 보존하기 위해 먼저 JSON 파싱 시도
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text}

            if resp.status_code != 200:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {str(data)[:500]}")
                # 재시도
                delay = (retry_backoff ** attempt) + (0.5 * attempt) + random.random()
                time.sleep(delay)
                continue

            # OpenAI 호환 형태 폴백 처리
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"invalid_response_shape(no choices): {str(data)[:500]}")

            ch0 = choices[0] or {}
            content = (ch0.get("message") or {}).get("content")
            if content is None:
                # 스트리밍 delta 형태 폴백
                content = (ch0.get("delta") or {}).get("content")

            if not content or not isinstance(content, str):
                raise RuntimeError(f"invalid_response_shape(no content): {str(data)[:500]}")

            return content
        except Exception as e:
            last_err = e
            delay = (retry_backoff ** attempt) + (0.5 * attempt) + random.random()
            time.sleep(delay)

    # 모든 재시도 실패
    raise last_err if last_err else RuntimeError("call_vllm: unknown error")

def strip_code_fences(s: str) -> str:
    """코드펜스 ```...``` (언어 토큰 대/소문자 무시) 제거"""
    t = s.strip()
    # 앞/뒤의 코드펜스를 느슨하게 제거
    t = re.sub(r"^```[\w-]*\s*", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    return t.strip()

def extract_json_block(text: str) -> str:
    """첫 '{'부터 마지막 '}'까지 슬라이스하여 JSON 본문만 추출"""
    t = strip_code_fences(text)
    s, e = t.find("{"), t.rfind("}")
    return t[s:e+1] if s != -1 and e != -1 and e > s else t


def _normalize_json_like(s: str) -> str:
    """주석/트레일링 콤마 등 경미한 위반을 정리 (유효 JSON으로 수선 시도)
    * 홑따옴표 강제 치환은 위험하므로 하지 않음
    """
    # 주석 제거 (// ... 또는 /* ... */)
    s = re.sub(r"//.*?$|/\*.*?\*/", "", s, flags=re.MULTILINE | re.DOTALL)
    # 트레일링 콤마 제거: }, ] 직전의 콤마
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def parse_llm_json(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    모델 출력에서 JSON을 파싱. 성공 여부, JSON 객체, 경고 문자열을 반환.
    1차: 코드펜스 제거 + {} 블록 슬라이스 후 표준 json.loads
    2차: 실패 시 주석/트레일링 콤마 제거 후 재시도
    파싱 이후에 한해 필요한 필드의 HTML escape 복원(\u003e → >)
    """
    warnings: List[str] = []
    raw = extract_json_block(text)

    try:
        obj = json.loads(raw)
    except Exception as e1:
        warnings.append(f"json_parse_error/pass1: {e1}")
        fixed = _normalize_json_like(raw)
        try:
            obj = json.loads(fixed)
        except Exception as e2:
            return False, {}, f"json_parse_error: {e2}"

    # 파싱 이후 필요한 필드에만 escape 복원
    try:
        for e in obj.get("edges", []) or []:
            if isinstance(e.get("type"), str):
                e["type"] = e["type"].replace("\\u003e", ">")
    except Exception as _:
        pass

    # warnings 필드 정규화 + 내부 경고 병합
    if "warnings" not in obj or not isinstance(obj.get("warnings"), list):
        obj["warnings"] = list(map(str, warnings)) if warnings else []
    else:
        obj["warnings"] = list(map(str, obj["warnings"])) + warnings

    return True, obj, "" if not warnings else ";".join(warnings)

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
    raw_dir = (Path(out_dir) / "raw"); raw_dir.mkdir(parents=True, exist_ok=True)

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
    if not files:
        print(f"[경고] Mermaid 폴더가 비어 있습니다: {mermaid_dir}")
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

        text = ""
        try:
            text = call_vllm(messages)
            # 원문 저장 (성공/실패 불문)
            raw_path = raw_dir / f"{name}_{model_name.replace('/', '_')}.txt"
            try:
                with open(raw_path, "w", encoding="utf-8") as rf:
                    rf.write(text)
            except Exception:
                pass
            ok, obj, warn = parse_llm_json(text)
        except Exception as e:
            ok, obj, warn = False, {}, f"request_error: {e}"
            # 실패해도 빈 원문 파일 남겨 디버깅에 활용
            raw_path = raw_dir / f"{name}_{model_name.replace('/', '_')}.txt"
            try:
                with open(raw_path, "w", encoding="utf-8") as rf:
                    rf.write(text or "")
            except Exception:
                pass

        # 요약 통계
        n_nodes_pred, n_edges_pred = (0, 0)
        warnings_str = ""
        if ok:
            n_nodes_pred, n_edges_pred = count_nodes_edges(obj)
            w = obj.get("warnings", [])
            if isinstance(w, list):
                warnings_str = ";".join(map(str, w))
            elif w is not None:
                warnings_str = str(w)
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
