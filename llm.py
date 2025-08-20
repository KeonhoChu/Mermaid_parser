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

# VLLM_ENDPOINT = "http://10.240.1.8:8012/v1/chat/completions"
# MODEL_NAME = "openai/gpt-oss-120b"

VLLM_ENDPOINT = "http://10.240.1.8:8011/v1/chat/completions"
MODEL_NAME = "skt/A.X-4.0-Light"

# -------- 프롬프트 (정규화) --------
SYSTEM_PROMPT = """Parse Mermaid flowchart to JSON only. No explanations. Start with { and end with }."""

# -------- 유틸: 호출/파싱 --------
def call_vllm(messages: List[Dict[str, str]],
              temperature: float = 0.0,
              top_p: float = 1.0,
              max_tokens: int = 7500,
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
    """더 강력한 JSON 추출 - 개선된 버전"""
    text = strip_code_fences(text)
    
    # 전체 텍스트가 이미 JSON인지 확인
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        return text_stripped
    
    # JSON이 텍스트 중간에 있을 수 있으므로 더 적극적으로 찾기
    json_candidates = []
    
    # 방법 1: 중괄호 매칭으로 완전한 JSON 찾기 (문자열 내부 고려)
    stack = []
    start_pos = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
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
                if not stack:
                    start_pos = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_pos != -1:
                        candidate = text[start_pos:i+1]
                        json_candidates.append(candidate)
    
    # 방법 2: 정규식으로 JSON 패턴 찾기 (개선된 버전)
    patterns = [
        r'\{[^{}]*"ok"[^{}]*\}',  # 단순한 JSON
        r'\{.*?"nodes".*?\}',     # nodes 포함
        r'\{.*?"edges".*?\}',     # edges 포함
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
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
    
    # 방법 3: 줄 단위로 JSON 시작/끝 찾기
    lines = text.split('\n')
    json_start = -1
    json_end = -1
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if json_start == -1 and line_stripped.startswith('{'):
            json_start = i
        if json_start != -1 and line_stripped.endswith('}') and '{' not in line_stripped:
            json_end = i
            break
    
    if json_start != -1 and json_end != -1:
        candidate = '\n'.join(lines[json_start:json_end+1])
        json_candidates.append(candidate)
    
    if json_candidates:
        # 후보들을 검증하고 가장 좋은 것 선택
        valid_candidates = []
        
        for candidate in json_candidates:
            # 기본 JSON 구조 확인
            if (candidate.count('{') == candidate.count('}') and 
                any(key in candidate for key in ['"ok"', '"nodes"', '"edges"']) and 
                len(candidate) > 50):
                valid_candidates.append(candidate)
        
        if valid_candidates:
            # 가장 긴 유효한 후보 선택
            return max(valid_candidates, key=len)
        else:
            # 유효한 후보가 없으면 가장 긴 후보 선택
            return max(json_candidates, key=len)
    
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
    """잘린 JSON을 복구하는 함수 - 완전히 재작성"""
    json_str = json_str.strip()
    
    if not json_str:
        return ""
    
    # 1단계: 역방향으로 완전한 구조 찾기
    lines = json_str.split('\n')
    
    # 마지막 줄부터 역순으로 검사
    valid_end_idx = len(lines)
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        
        # 빈 줄은 스킵
        if not line:
            continue
        
        # 명백히 불완전한 줄 패턴들
        incomplete_patterns = [
            # 따옴표가 시작되었지만 끝나지 않음
            r'"[^"]*$',
            # 키:값 구조가 불완전
            r'"[^"]*":\s*"[^"]*$',
            # 객체나 배열이 시작되었지만 끝나지 않음
            r'.*[{\[]$',
        ]
        
        is_incomplete = any(re.search(pattern, line) for pattern in incomplete_patterns)
        
        # 따옴표 개수가 홀수인지 확인 (문자열 내부 고려)
        quote_count = line.count('"')
        has_unclosed_quotes = quote_count % 2 == 1
        
        if is_incomplete or has_unclosed_quotes:
            # 이 줄은 불완전하므로 제외
            valid_end_idx = i
        else:
            # 완전한 줄을 찾았으므로 중단
            break
    
    # 유효한 줄들만 사용
    if valid_end_idx < len(lines):
        lines = lines[:valid_end_idx]
    
    if not lines:
        return ""
    
    result = '\n'.join(lines)
    
    # 2단계: 구조적 완성도 검사 및 수정
    try:
        # 간단한 파싱 테스트
        json.loads(result)
        return result  # 이미 완전한 JSON
    except json.JSONDecodeError:
        pass
    
    # 3단계: 구조 복구
    result = result.strip()
    
    # 트레일링 콤마 제거
    result = re.sub(r',\s*$', '', result)
    
    # 불완전한 마지막 요소 제거 (키만 있고 값이 없는 경우)
    lines = result.split('\n')
    if lines:
        last_line = lines[-1].strip()
        # "key": 형태로 끝나는 경우 제거
        if re.match(r'^\s*"[^"]*":\s*$', last_line):
            lines = lines[:-1]
            result = '\n'.join(lines)
            result = re.sub(r',\s*$', '', result)
    
    # 4단계: 괄호 균형 맞추기
    open_braces = result.count('{') - result.count('}')
    open_brackets = result.count('[') - result.count(']')
    
    if open_braces > 0 or open_brackets > 0:
        # 배열 먼저 닫기
        for _ in range(open_brackets):
            result += '\n    ]'
        
        # 객체 닫기  
        for _ in range(open_braces):
            result += '\n  }'
    
    # 5단계: 최종 검증 및 폴백
    try:
        json.loads(result)
        return result
    except json.JSONDecodeError:
        # 여전히 파싱이 안 되면 최소한의 유효한 JSON 구조 만들기
        return '{"ok": false, "error": "truncated JSON recovery failed"}'

def create_fallback_json(text: str, reason: str) -> Dict[str, Any]:
    """파싱 실패 시 기본 JSON 구조 생성"""
    return {
        "ok": False,
        "warnings": [f"JSON 파싱 실패: {reason}"],
        "nodes": [],
        "edges": [],
        "subgraphs": [],
        "decisions": [],
        "summary": {
            "natural_language": f"파싱 실패: {reason}",
            "entry_nodes": [],
            "exit_nodes": []
        },
        "raw_response": text[:500] + "..." if len(text) > 500 else text
    }

def parse_llm_json(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """강화된 JSON 파싱 - 개선된 폴백 메커니즘"""
    warnings = []
    
    if not text.strip():
        return True, create_fallback_json("", "빈 응답"), "빈 응답"
    
    # 모델이 분석 텍스트를 생성했는지 감지 (더 포괄적으로)
    text_indicators = [
        'We need to parse', 'Let me analyze', 'Looking at this', 
        'This flowchart', 'The diagram', 'Based on the'
    ]
    
    if not text.strip().startswith('{') and any(indicator in text for indicator in text_indicators):
        warnings.append("모델이 분석 텍스트를 생성함")
        return True, create_fallback_json(text, "텍스트 분석 응답"), ";".join(warnings)
    
    # 1차: 원본 텍스트를 직접 JSON으로 파싱 시도
    try:
        obj = json.loads(text.strip())
        warnings.append("원본 텍스트 직접 파싱 성공")
    except json.JSONDecodeError as e:
        print(f"원본 텍스트 파싱 실패: {e}")
        
        # 2차: 강화된 추출 및 파싱
        try:
            json_text = extract_json_robustly(text)
            if not json_text or json_text == text.strip():
                raise ValueError("JSON 추출 실패")
                
            obj = json.loads(json_text)
            warnings.append("JSON 추출 후 파싱 성공")
        except (json.JSONDecodeError, ValueError) as e1:
            warnings.append(f"JSON 추출 후 파싱 실패: {e1}")
            
            # 3차: 구문 수정 후 재시도
            try:
                json_text = extract_json_robustly(text)
                fixed_json = fix_json_syntax(json_text)
                if not fixed_json:
                    raise ValueError("JSON 구문 수정 실패")
                    
                obj = json.loads(fixed_json)
                warnings.append("JSON 구문 수정 후 파싱 성공")
            except (json.JSONDecodeError, ValueError) as e2:
                warnings.append(f"구문 수정 후 파싱 실패: {e2}")
                
                # 4차: 잘린 JSON 복구 시도
                try:
                    truncated_json = fix_truncated_json(fixed_json if 'fixed_json' in locals() else json_text if 'json_text' in locals() else text)
                    if not truncated_json:
                        raise ValueError("JSON 복구 실패")
                        
                    obj = json.loads(truncated_json)
                    warnings.append("잘린 JSON 복구 후 파싱 성공")
                except (json.JSONDecodeError, ValueError) as e3:
                    warnings.append(f"잘린 JSON 복구 실패: {e3}")
                    
                    # 5차: 완전한 폴백 - 기본 구조 생성
                    print(f"모든 JSON 파싱 방법 실패. 폴백 JSON 생성")
                    return True, create_fallback_json(text, f"모든 파싱 방법 실패: {e3}"), ";".join(warnings)
                except Exception as e3:
                    warnings.append(f"JSON 복구 예외: {e3}")
                    return True, create_fallback_json(text, f"복구 예외: {e3}"), ";".join(warnings)
    
    # JSON 검증 및 보완
    try:
        # 필수 필드 확인 및 추가
        if not isinstance(obj, dict):
            obj = {"parsed_data": obj}
        
        # 기본 필드들이 없으면 추가
        required_fields = {
            "ok": True,
            "warnings": [],
            "nodes": [],
            "edges": [],
            "subgraphs": [],
            "decisions": []
        }
        
        for field, default_value in required_fields.items():
            if field not in obj:
                obj[field] = default_value
        
        # summary 필드 확인
        if "summary" not in obj:
            obj["summary"] = {
                "natural_language": "자동 생성된 요약",
                "entry_nodes": [],
                "exit_nodes": []
            }
        
        # escape 복원
        for edge in obj.get("edges", []):
            if isinstance(edge.get("type"), str):
                edge["type"] = edge["type"].replace("\\u003e", ">")
    except Exception as e:
        warnings.append(f"JSON 후처리 실패: {e}")
        return True, create_fallback_json(text, f"후처리 실패: {e}"), ";".join(warnings)
    
    # warnings 필드 정규화
    existing_warnings = obj.get("warnings", [])
    if not isinstance(existing_warnings, list):
        existing_warnings = [str(existing_warnings)] if existing_warnings else []
    
    obj["warnings"] = existing_warnings + warnings
    
    warning_msg = ";".join(warnings) if warnings else ""
    return True, obj, warning_msg

def count_nodes_edges_subgraphs(obj: Dict[str, Any]) -> Tuple[int, int, int]:
    """노드, 엣지, 서브그래프 개수 세기"""
    try:
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])
        subgraphs = obj.get("subgraphs", [])
        return (
            len(nodes) if isinstance(nodes, list) else 0,
            len(edges) if isinstance(edges, list) else 0,
            len(subgraphs) if isinstance(subgraphs, list) else 0
        )
    except Exception:
        return 0, 0, 0

# -------- 배치 실행 --------
def run_batch(mermaid_dir: str = "mermaid",
              out_dir: str = "results",
              level_csv: Optional[str] = "mermaid_level_analysis_results.csv",
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
            with open(level_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    level_map[str(row["name"])] = {
                        "level": row.get("level"),
                        "N": row.get("N"),
                        "E": row.get("E"),
                        "SUBGRAPH_COUNT": row.get("SUBGRAPH_COUNT"),
                        "explanations": row.get("explanations")
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
        
        # 프롬프트 준비 (토큰 최적화)
        user_msg = f"""Parse this Mermaid flowchart to JSON:

{mermaid_code}

Return only JSON:
{{
  "ok": true,
  "nodes": [{{"id": "node_id", "label": "node_label", "shape": "rect"}}],
  "edges": [{{"source": "node1", "target": "node2", "type": "-->"}}],
  "subgraphs": [{{"id": "subgraph_id", "title": "title"}}],
  "summary": {{"natural_language": "Korean description"}}
}}

Shapes: []=rect, {{}}=diamond, ()=circle. Start with {{"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        # API 호출
        llm_text = ""
        try:
            # 입력 토큰 수 추정 (보수적: 2글자 = 1토큰, 한글/특수문자 고려)
            input_tokens = len(user_msg + SYSTEM_PROMPT) // 2
            max_output_tokens = min(6000, 8000 - input_tokens - 500)  # 500토큰 여유분
            
            print(f"추정 입력토큰: {input_tokens}, 출력토큰: {max_output_tokens}")
            
            if max_output_tokens < 1000:
                print(f"입력이 너무 큽니다. 파일 크기: {len(mermaid_code)}글자")
                max_output_tokens = 1000  # 최소한의 출력 보장
                
            llm_text = call_vllm(messages, temperature=0.0, top_p=0.9, max_tokens=max_output_tokens)
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
        n_nodes, n_edges, n_subgraphs = count_nodes_edges_subgraphs(obj) if ok else (0, 0, 0)
        
        # 레벨 정보 가져오기
        level_info = level_map.get(name, {})
        
        # 결과 행 추가
        rows.append({
            "name": name,
            "model": model_name,
            "level": level_info.get("level", ""),
            "N_actual": level_info.get("N", ""),
            "E_actual": level_info.get("E", ""),
            "SUBGRAPH_COUNT_actual": level_info.get("SUBGRAPH_COUNT", ""),
            "explanations": level_info.get("explanations", ""),
            "llm_ok": ok,
            "llm_warnings": warnings,
            "n_nodes_pred": n_nodes,
            "n_edges_pred": n_edges,
            "n_subgraphs_pred": n_subgraphs,
            "json_raw_path": str(json_path)
        })
        
        print(f"노드: {n_nodes}, 엣지: {n_edges}, 서브그래프: {n_subgraphs}")
    
    # CSV 저장
    out_csv = Path(out_dir) / "SKT.csv"
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "name", "model", "level", "N_actual", "E_actual", "SUBGRAPH_COUNT_actual", "explanations",
                "llm_ok", "llm_warnings", "n_nodes_pred", "n_edges_pred", "n_subgraphs_pred", "json_raw_path"
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
        print(f"연결 성공!")
        print(f"응답: {response[:200]}...")
    except Exception as e:
        print(f"연결 실패: {e}")

if __name__ == "__main__":
    # 연결 테스트 먼저 실행
    test_connection()
    print("\n" + "="*50 + "\n")
    
    # 배치 실행
    run_batch(
        mermaid_dir="mermaid",
        out_dir="results",
        level_csv="mermaid_level_analysis_results.csv",
        model_name=MODEL_NAME
    )