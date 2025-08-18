# vLLM Chat Completions 배치 러너 (OpenAI 호환) - 수정된 버전
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
SYSTEM_PROMPT = """You are a JSON-only Mermaid diagram parser. You MUST respond with only valid JSON, no explanations or analysis.

Your task: Parse the Mermaid flowchart and output JSON with the exact schema provided.

CRITICAL RULES:
1. Output ONLY JSON - no text before or after
2. Start response with { and end with }
3. Do not explain your reasoning
4. Do not analyze the diagram in text
5. Just extract nodes, edges, and subgraphs into the JSON format
"""

# -------- 유틸: 호출/파싱 --------
def call_vllm(messages: List[Dict[str, str]],
              temperature: float = 0.0,
              top_p: float = 1.0,
              max_tokens: int = 16000,
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
    }
    
    # seed는 서버가 지원하지 않을 수 있으므로 조건부 추가
    if seed is not None:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[시도 {attempt}/{retries}] API 호출 중...")
            resp = requests.post(VLLM_ENDPOINT, json=payload, timeout=timeout, headers=headers)
            
            # 응답 상태 확인
            print(f"응답 상태: {resp.status_code}")
            
            # JSON 파싱 시도
            try:
                data = resp.json()
            except Exception as e:
                print(f"JSON 파싱 실패: {e}")
                data = {"raw_text": resp.text}

            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}"
                if isinstance(data, dict) and "error" in data:
                    error_msg += f": {data['error']}"
                else:
                    error_msg += f": {str(data)[:500]}"
                last_err = RuntimeError(error_msg)
                print(f"에러 응답: {error_msg}")
                
                # 재시도 전 대기
                if attempt < retries:
                    delay = (retry_backoff ** attempt) + (0.5 * attempt) + random.uniform(0, 1)
                    print(f"{delay:.2f}초 후 재시도...")
                    time.sleep(delay)
                continue

            # OpenAI 호환 응답 파싱
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"응답에 choices가 없음: {str(data)[:500]}")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")
            
            # reasoning 모델의 경우 reasoning_content를 사용
            if content is None:
                content = message.get("reasoning_content")
            
            if content is None:
                # 스트리밍 형태 폴백
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content is None:
                    content = delta.get("reasoning_content")

            if not content or not isinstance(content, str):
                raise RuntimeError(f"유효하지 않은 content: {str(data)[:500]}")

            print(f"응답 길이: {len(content)} 문자")
            return content
            
        except requests.exceptions.Timeout:
            last_err = RuntimeError("요청 타임아웃")
            print(f"타임아웃 발생 (시도 {attempt})")
        except requests.exceptions.ConnectionError:
            last_err = RuntimeError("연결 실패")
            print(f"연결 실패 (시도 {attempt})")
        except Exception as e:
            last_err = e
            print(f"예외 발생 (시도 {attempt}): {e}")
        
        # 재시도 전 대기
        if attempt < retries:
            delay = (retry_backoff ** attempt) + (0.5 * attempt) + random.uniform(0, 1)
            print(f"{delay:.2f}초 후 재시도...")
            time.sleep(delay)

    # 모든 재시도 실패
    raise last_err if last_err else RuntimeError("알 수 없는 오류")

def strip_code_fences(s: str) -> str:
    """코드펜스 제거"""
    t = s.strip()
    # 시작 코드펜스 제거
    t = re.sub(r"^```[\w-]*\s*", "", t, flags=re.IGNORECASE | re.MULTILINE)
    # 끝 코드펜스 제거
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    return t.strip()

def extract_json_robustly(text: str) -> str:
    """더 강력한 JSON 추출"""
    text = strip_code_fences(text)
    
    # 전체 텍스트가 이미 JSON인지 확인
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        return text_stripped
    
    # JSON이 텍스트 중간에 있을 수 있으므로 더 적극적으로 찾기
    json_candidates = []
    
    # 방법 1: 중괄호 매칭으로 완전한 JSON 찾기
    stack = []
    start_pos = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_pos = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_pos != -1:
                    candidate = text[start_pos:i+1]
                    json_candidates.append(candidate)
    
    # 방법 2: "ok"나 "nodes", "edges" 키워드가 포함된 JSON 찾기 (개선된 버전)
    for match in re.finditer(r'\{[^}]*(?:"ok"|"nodes"|"edges")', text, re.DOTALL):
        start_idx = match.start()
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start_idx:i+1]
                        json_candidates.append(candidate)
                        break
    
    if json_candidates:
        # 가장 긴 후보 선택 (보통 완전한 응답)
        best_candidate = max(json_candidates, key=len)
        
        # 추가 검증: 기본 JSON 키들이 있는지 확인
        for candidate in sorted(json_candidates, key=len, reverse=True):
            if any(key in candidate for key in ['"ok"', '"nodes"', '"edges"']) and len(candidate) > 100:
                return candidate
        
        return best_candidate
    
    # 폴백: 첫 번째와 마지막 중괄호 사이
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    return text.strip()

def fix_json_syntax(s: str) -> str:
    """JSON 구문 오류 수정 - 강화된 버전"""
    # 1. 주석 제거 (한 줄 주석)
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    # 블록 주석 제거
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    
    # 2. 불완전한 JSON 완성 (맨 끝이 잘린 경우)
    s = s.strip()
    
    # 3. 줄바꿈으로 잘린 문자열 수정
    lines = s.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # 마지막 문자가 콤마나 따옴표가 아니고, 다음 줄이 있다면
        if (i < len(lines) - 1 and 
            line and 
            not line.endswith((',', '"', '}', ']', '{', '['))):
            
            # 문자열이 따옴표로 시작했지만 끝나지 않은 경우
            if '"' in line and line.count('"') % 2 == 1:
                # 다음 줄의 시작이 문자열 계속인지 확인
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith(('"', '}', ']', ',')):
                    # 현재 줄과 다음 줄을 합침
                    lines[i + 1] = line + ' ' + next_line
                    continue
        
        fixed_lines.append(line)
    
    s = '\n'.join(fixed_lines)
    
    # 4. 트레일링 콤마 제거
    s = re.sub(r',\s*([}\]])', r'\1', s)
    
    # 5. 키 이름에 따옴표 추가 (JavaScript 스타일 → JSON)
    s = re.sub(r'(\w+):\s*', r'"\1": ', s)
    
    # 6. 이미 따옴표가 있는 키는 중복 제거
    s = re.sub(r'""(\w+)":', r'"\1":', s)
    
    # 7. 값에 따옴표 없는 문자열 수정
    def fix_unquoted_strings(match):
        value = match.group(1).strip()
        if value in ['true', 'false', 'null']:
            return f': {value}'
        elif value.replace('.', '').replace('-', '').isdigit():
            return f': {value}'
        elif not value.startswith('"'):
            return f': "{value}"'
        return f': {value}'
    
    s = re.sub(r':\s*([^",\[\]{}]+?)(?=\s*[,}\]])', fix_unquoted_strings, s)
    
    # 8. 불완전한 문자열 수정 (따옴표 매칭)
    lines = s.split('\n')
    fixed_lines = []
    
    for line in lines:
        if ':' in line and '"' in line:
            quote_count = line.count('"')
            # 홀수 개의 따옴표가 있으면서 줄이 불완전하게 끝난 경우
            if (quote_count % 2 == 1 and 
                not line.strip().endswith(('}', ']', ',', '"')) and
                ':' in line):
                line = line.rstrip() + '"'
        fixed_lines.append(line)
    
    s = '\n'.join(fixed_lines)
    
    # 9. 불완전한 객체/배열 완성
    s = s.strip()
    if s.startswith('{') and not s.endswith('}'):
        open_braces = s.count('{')
        close_braces = s.count('}')
        s += '}' * (open_braces - close_braces)
    
    if s.startswith('[') and not s.endswith(']'):
        open_brackets = s.count('[')
        close_brackets = s.count(']')
        s += ']' * (open_brackets - close_brackets)
    
    # 10. 마지막 콤마 처리 (다시 한번)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    
    # 11. 빈 배열/객체 뒤의 콤마 제거
    s = re.sub(r'([\[\{])\s*,', r'\1', s)
    
    # 12. 연속된 콤마 제거
    s = re.sub(r',\s*,', ',', s)
    
    return s

def fix_truncated_json(json_str: str) -> str:
    """잘린 JSON을 복구하는 함수"""
    json_str = json_str.strip()
    
    # 마지막 완전한 구조를 찾기 위해 역순으로 탐색
    brace_stack = []
    last_complete_pos = -1
    
    for i, char in enumerate(json_str):
        if char == '{':
            brace_stack.append('{')
        elif char == '}':
            if brace_stack and brace_stack[-1] == '{':
                brace_stack.pop()
                if not brace_stack:  # 완전한 객체 완성
                    last_complete_pos = i
        elif char == '[':
            brace_stack.append('[')
        elif char == ']':
            if brace_stack and brace_stack[-1] == '[':
                brace_stack.pop()
    
    if last_complete_pos > 0:
        # 마지막 완전한 위치까지만 사용
        return json_str[:last_complete_pos + 1]
    
    # 완전한 구조를 찾지 못한 경우, 최소한의 구조라도 만들어봄
    if json_str.startswith('{'):
        # 마지막 콤마 뒤에 있는 불완전한 항목 제거
        lines = json_str.split('\n')
        complete_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('}', ']', ',', '"')):
                # 불완전한 줄이므로 제외
                break
            complete_lines.append(line)
        
        result = '\n'.join(complete_lines)
        
        # 마지막이 콤마로 끝나면 제거
        result = re.sub(r',\s*$', '', result.strip())
        
        # 중괄호 닫기
        if not result.endswith('}'):
            result += '}'
            
        return result
    
    return json_str

def parse_llm_json(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """강화된 JSON 파싱"""
    warnings = []
    
    if not text.strip():
        return False, {}, "빈 응답"
    
    # 모델이 분석 텍스트를 생성했는지 감지
    if not text.strip().startswith('{') and 'We need to parse' in text:
        warnings.append("모델이 분석 텍스트를 생성함 - JSON 형식 변환 시도")
        # 간단한 폴백 JSON 생성
        fallback_json = {
            "ok": False,
            "warnings": ["모델이 JSON 대신 분석 텍스트를 생성했습니다"],
            "nodes": [],
            "edges": [],
            "subgraphs": [],
            "decisions": [],
            "summary": {
                "natural_language": "파싱 실패: 텍스트 분석 응답",
                "entry_nodes": [],
                "exit_nodes": []
            }
        }
        return True, fallback_json, ";".join(warnings)
    
    # 1차: 원본 텍스트를 직접 JSON으로 파싱 시도
    try:
        obj = json.loads(text.strip())
        warnings.append("원본 텍스트 직접 파싱 성공")
    except json.JSONDecodeError as e:
        print(f"원본 텍스트 파싱 실패: {e}")
        
        # 2차: 강화된 추출 및 파싱
        json_text = extract_json_robustly(text)
        
        try:
            obj = json.loads(json_text)
            warnings.append("JSON 추출 후 파싱 성공")
        except json.JSONDecodeError as e1:
            warnings.append(f"JSON 추출 후 파싱 실패: {e1}")
            print(f"추출된 JSON 길이: {len(json_text)}")
            print(f"추출된 JSON 시작: {json_text[:200]}...")
            
            # 3차: 구문 수정 후 재시도
            try:
                fixed_json = fix_json_syntax(json_text)
                obj = json.loads(fixed_json)
                warnings.append("JSON 구문 수정 후 파싱 성공")
            except json.JSONDecodeError as e2:
                warnings.append(f"구문 수정 후 파싱 실패: {e2}")
                print(f"수정된 JSON 길이: {len(fixed_json)}")
                print(f"수정된 JSON 시작: {fixed_json[:200]}...")
                
                # 4차: 잘린 JSON 복구 시도
                try:
                    # JSON이 중간에 잘렸을 가능성이 높으므로 마지막 완전한 객체/배열까지만 파싱
                    truncated_json = fix_truncated_json(fixed_json)
                    obj = json.loads(truncated_json)
                    warnings.append("잘린 JSON 복구 후 파싱 성공")
                except json.JSONDecodeError as e3:
                    warnings.append(f"잘린 JSON 복구 실패: {e3}")
                    return False, {}, f"JSON 파싱 실패: {e3}"
                except Exception as e3:
                    return False, {}, f"JSON 복구 실패: {e3}"
    
    # escape 복원
    try:
        for edge in obj.get("edges", []):
            if isinstance(edge.get("type"), str):
                edge["type"] = edge["type"].replace("\\u003e", ">")
    except Exception:
        pass
    
    # warnings 필드 정규화
    existing_warnings = obj.get("warnings", [])
    if not isinstance(existing_warnings, list):
        existing_warnings = [str(existing_warnings)] if existing_warnings else []
    
    obj["warnings"] = existing_warnings + warnings
    
    warning_msg = ";".join(warnings) if warnings else ""
    return True, obj, warning_msg

def count_nodes_edges(obj: Dict[str, Any]) -> Tuple[int, int]:
    """노드와 엣지 개수 세기"""
    try:
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])
        return len(nodes) if isinstance(nodes, list) else 0, len(edges) if isinstance(edges, list) else 0
    except Exception:
        return 0, 0

# -------- 배치 실행 --------
def run_batch(mermaid_dir: str = "mermaid",
              out_dir: str = "results",
              level_csv: Optional[str] = "mermaid_analysis_results.csv",
              model_name: str = MODEL_NAME):
    """배치 실행 메인 함수"""
    
    print(f"배치 실행 시작:")
    print(f"  - Mermaid 디렉토리: {mermaid_dir}")
    print(f"  - 출력 디렉토리: {out_dir}")
    print(f"  - 모델: {model_name}")
    print(f"  - 엔드포인트: {VLLM_ENDPOINT}")
    
    # 디렉토리 생성
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_dir = Path(out_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 레벨 CSV 로드
    level_map: Dict[str, Dict[str, Any]] = {}
    if level_csv and Path(level_csv).exists():
        try:
            import pandas as pd
            df = pd.read_csv(level_csv)
            for _, row in df.iterrows():
                level_map[str(row["name"])] = {
                    "level": row.get("level"),
                    "score_final": row.get("score_final")
                }
            print(f"레벨 CSV 로드 완료: {len(level_map)}개 항목")
        except Exception as e:
            print(f"레벨 CSV 로드 실패: {e}")
    
    # Mermaid 파일 찾기
    pattern = os.path.join(mermaid_dir, "*.mmd")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"[경고] Mermaid 파일을 찾을 수 없습니다: {pattern}")
        return
    
    print(f"처리할 파일: {len(files)}개")
    
    # 연결 테스트
    print("\n연결 테스트 중...")
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, please respond with just 'OK' to test the connection."}
    ]
    
    try:
        test_response = call_vllm(test_messages, max_tokens=50)
        print(f"연결 테스트 성공: {test_response[:50]}...")
    except Exception as e:
        print(f"연결 테스트 실패: {e}")
        print("서버 상태를 확인하세요.")
        return
    
    # 파일 처리
    rows = []
    
    for i, filepath in enumerate(files, 1):
        name = Path(filepath).stem
        print(f"\n[{i}/{len(files)}] 처리 중: {name}")
        
        # Mermaid 파일 읽기
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mermaid_code = f.read().strip()
        except Exception as e:
            print(f"파일 읽기 실패: {e}")
            continue
        
        if not mermaid_code:
            print("빈 파일 스킵")
            continue
        
        # 프롬프트 준비
        user_msg = f"""Parse this Mermaid flowchart and return JSON ONLY:

{mermaid_code}

Output JSON with this exact structure (no other text):

{{
  "ok": true,
  "warnings": [],
  "nodes": [
    {{
      "id": "node_id",
      "label": "node_label",
      "shape": "rect",
      "subgraphs": []
    }}
  ],
  "edges": [
    {{
      "source": "node1",
      "target": "node2", 
      "type": "-->",
      "label": null
    }}
  ],
  "subgraphs": [
    {{
      "id": "subgraph_id",
      "title": "subgraph_title"
    }}
  ],
  "decisions": [],
  "summary": {{
    "natural_language": "Brief description",
    "entry_nodes": ["START"],
    "exit_nodes": ["END"]
  }}
}}

Shape mapping: [] = "rect", {{}} = "diamond", () = "circle", (()) = "double_circle", [[]] = "double_rect", >> = "subroutine"

START YOUR RESPONSE WITH {{ - NO OTHER TEXT"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        # API 호출
        llm_text = ""
        try:
            # JSON 출력을 위해 더 제한적인 파라미터 사용
            llm_text = call_vllm(messages, temperature=0.0, top_p=0.9, max_tokens=3000)
            ok, obj, warnings = parse_llm_json(llm_text)
            print(f"파싱 결과: {'성공' if ok else '실패'}")
            if warnings:
                print(f"경고: {warnings}")
        except Exception as e:
            ok, obj, warnings = False, {}, f"API 호출 실패: {e}"
            print(f"API 호출 실패: {e}")
        
        # 원문 저장
        raw_path = raw_dir / f"{name}_{model_name.replace('/', '_')}.txt"
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(llm_text)
        except Exception as e:
            print(f"원문 저장 실패: {e}")
        
        # JSON 저장
        json_path = json_dir / f"{name}_{model_name.replace('/', '_')}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                if ok:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
                else:
                    json.dump({
                        "error": warnings,
                        "raw_response": llm_text
                    }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"JSON 저장 실패: {e}")
        
        # 통계 수집
        n_nodes, n_edges = count_nodes_edges(obj) if ok else (0, 0)
        
        # 레벨 정보 가져오기
        level_info = level_map.get(name, {})
        
        # 결과 행 추가
        rows.append({
            "name": name,
            "model": model_name,
            "level": level_info.get("level", ""),
            "score_final": level_info.get("score_final", ""),
            "llm_ok": ok,
            "llm_warnings": warnings,
            "n_nodes_pred": n_nodes,
            "n_edges_pred": n_edges,
            "json_raw_path": str(json_path)
        })
        
        print(f"노드: {n_nodes}, 엣지: {n_edges}")
    
    # CSV 저장
    out_csv = Path(out_dir) / "llm_runs.csv"
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "name", "model", "level", "score_final",
                "llm_ok", "llm_warnings", "n_nodes_pred", "n_edges_pred", "json_raw_path"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[완료] 결과 CSV 저장: {out_csv}")
    except Exception as e:
        print(f"CSV 저장 실패: {e}")

def test_connection():
    """연결 테스트 함수"""
    print(f"vLLM 서버 연결 테스트: {VLLM_ENDPOINT}")
    print(f"모델: {MODEL_NAME}")
    
    try:
        response = call_vllm([
            {"role": "user", "content": "Hello"}
        ], max_tokens=50)
        print(f"✅ 연결 성공!")
        print(f"응답: {response[:200]}...")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

if __name__ == "__main__":
    # 연결 테스트 먼저 실행
    test_connection()
    print("\n" + "="*50 + "\n")
    
    # 배치 실행
    run_batch(
        mermaid_dir="mermaid",
        out_dir="results",
        level_csv="mermaid_analysis_results.csv",
        model_name=MODEL_NAME
    )