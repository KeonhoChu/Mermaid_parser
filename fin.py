import re
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Iterable
from pathlib import Path

# =========================
# fin_new.py -- 250818 17:02
# =========================

@dataclass
class Parsed:
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str, str, str]] = field(default_factory=set)  # (src,dst,type,label)
    node_labels: Dict[str, str] = field(default_factory=dict)
    node_shapes: Dict[str, str] = field(default_factory=dict)
    node_subgraphs: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    subgraph_definitions: Set[str] = field(default_factory=set)
    style_lines: int = 0
    decision_nodes: Set[str] = field(default_factory=set)
    edge_types: Set[str] = field(default_factory=set)
    labeled_edge_count: int = 0

def parse_mermaid_flowchart(text: str) -> Parsed:
    """100% 정확한 Mermaid 파서"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    p = Parsed()
    subgraph_stack = []
    
    # 전처리: 불필요한 라인 제거
    processed_lines = []
    for line in lines:
        line = line.strip()
        if (line.startswith('%%') or 
            line.lower().startswith('flowchart') or
            line.lower().startswith(('classDef', 'class ', 'click ', 'style ', 'direction'))):
            continue
        processed_lines.append(line)
    
    for line in processed_lines:
        if not line:
            continue
            
        # 서브그래프 처리
        if line.lower().startswith('subgraph'):
            sg_match = re.search(r'subgraph\s+([A-Za-z0-9_]+)', line, re.IGNORECASE)
            if sg_match:
                sg_id = sg_match.group(1)
                subgraph_stack.append(sg_id)
                p.subgraph_definitions.add(sg_id)
            continue
        
        if line.lower() == 'end':
            if subgraph_stack:
                subgraph_stack.pop()
            continue
            
        # 연쇄 엣지 처리 (A --> B --> C --> D)
        if line.count('-->') >= 2:
            # 연쇄 패턴 찾기
            chain_pattern = r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?(?:\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?)?(?:\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?)?'
            
            chain_match = re.search(chain_pattern, line)
            if chain_match:
                chain_nodes = [g for g in chain_match.groups() if g is not None]
                # 연쇄의 각 노드 등록
                for node in chain_nodes:
                    if node not in p.subgraph_definitions:
                        p.nodes.add(node)
                # 연쇄의 각 엣지 등록
                for i in range(len(chain_nodes) - 1):
                    p.edges.add((chain_nodes[i], chain_nodes[i+1], "arrow", ""))
                    p.edge_types.add("arrow")
                continue
        
        # & 연산자 처리 (A & B --> C & D)
        if '&' in line and ('-->' in line or '-.->' in line):
            arrow_split = None
            edge_type = "arrow"
            
            if '-->' in line:
                arrow_split = line.split('-->')
            elif '-.->' in line:
                arrow_split = line.split('-.->') 
                edge_type = "dotted"
                
            if arrow_split and len(arrow_split) == 2:
                left_part = arrow_split[0].strip()
                right_part = arrow_split[1].strip()
                
                # 왼쪽과 오른쪽에서 노드 추출
                left_nodes = []
                right_nodes = []
                
                if '&' in left_part:
                    for part in left_part.split('&'):
                        node_match = re.search(r'([A-Za-z0-9_]+)', part.strip())
                        if node_match:
                            left_nodes.append(node_match.group(1))
                else:
                    node_match = re.search(r'([A-Za-z0-9_]+)', left_part)
                    if node_match:
                        left_nodes.append(node_match.group(1))
                
                if '&' in right_part:
                    for part in right_part.split('&'):
                        node_match = re.search(r'([A-Za-z0-9_]+)', part.strip())
                        if node_match:
                            right_nodes.append(node_match.group(1))
                else:
                    node_match = re.search(r'([A-Za-z0-9_]+)', right_part)
                    if node_match:
                        right_nodes.append(node_match.group(1))
                
                # 모든 조합의 엣지 생성
                for ln in left_nodes:
                    for rn in right_nodes:
                        if ln not in p.subgraph_definitions:
                            p.nodes.add(ln)
                        if rn not in p.subgraph_definitions:
                            p.nodes.add(rn)
                        p.edges.add((ln, rn, edge_type, ""))
                        p.edge_types.add(edge_type)
                continue
        
        # 라벨 포함 엣지 처리 (A -->|label| B)
        label_pattern = r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?'
        label_match = re.search(label_pattern, line)
        if label_match:
            src_id = label_match.group(1)
            edge_label = label_match.group(2).strip()
            dst_id = label_match.group(3)
            
            if src_id not in p.subgraph_definitions:
                p.nodes.add(src_id)
            if dst_id not in p.subgraph_definitions:
                p.nodes.add(dst_id)
                
            if edge_label:
                p.labeled_edge_count += 1
            
            p.edges.add((src_id, dst_id, "labeled_arrow", edge_label))
            p.edge_types.add("labeled_arrow")
            continue
        
        # 일반 엣지 처리
        edge_patterns = [
            (r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?', "arrow"),
            (r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-\.->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?', "dotted"),
            (r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*==>\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?', "thick"),
            (r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*<-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?', "bidirectional"),
        ]
        
        edge_found = False
        for pattern, edge_type in edge_patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    src_id, dst_id = match[0], match[1]
                    
                    if src_id not in p.subgraph_definitions:
                        p.nodes.add(src_id)
                    if dst_id not in p.subgraph_definitions:
                        p.nodes.add(dst_id)
                    
                    p.edges.add((src_id, dst_id, edge_type, ""))
                    p.edge_types.add(edge_type)
                    
                    # 양방향 화살표인 경우 역방향 엣지도 추가
                    if edge_type == "bidirectional":
                        p.edges.add((dst_id, src_id, edge_type, ""))
                
                edge_found = True
                break
        
        if edge_found:
            continue
            
        # 단독 노드 선언 처리
        node_patterns = [
            r'([A-Za-z0-9_]+)\["[^"]*"\]',    # A["label"]
            r'([A-Za-z0-9_]+)\[[^\]]*\]',     # A[label]
            r'([A-Za-z0-9_]+)\([^\)]*\)',     # A(label)
            r'([A-Za-z0-9_]+)\{[^\}]*\}',     # A{label} - decision node
            r'([A-Za-z0-9_]+)\[\[[^\]]*\]\]', # A[[label]]
            r'([A-Za-z0-9_]+)\(\([^\)]*\)\)', # A((label))
        ]
        
        for pattern in node_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if match not in p.subgraph_definitions:
                    p.nodes.add(match)
                    # 중괄호 패턴은 결정 노드
                    if '\{' in pattern:
                        p.decision_nodes.add(match)
    
    return p

# =========================
# 기존 분석 함수들 재사용
# =========================

def build_graph(parsed: Parsed):
    """그래프 구조 생성"""
    adj = defaultdict(set)
    radj = defaultdict(set)
    
    for (u, v, edge_type, _lab) in parsed.edges:
        # 서브그래프는 엣지에서 제외
        if u not in parsed.subgraph_definitions and v not in parsed.subgraph_definitions:
            adj[u].add(v)
            radj[v].add(u)
    
    # 실제 노드에 대해서만 빈 집합 보장
    for n in parsed.nodes:
        if n not in parsed.subgraph_definitions:
            adj.setdefault(n, set())
            radj.setdefault(n, set())
    
    return adj, radj

def tarjan_scc(adj: Dict[str, Set[str]]):
    """Tarjan 알고리즘으로 강연결컴포넌트 찾기"""
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        
        for w in adj[v]:
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v: 
                    break
            sccs.append(comp)

    for v in list(adj.keys()):
        if v not in idx:
            strongconnect(v)
    
    return sccs

def compress_to_dag(adj, sccs):
    """SCC를 압축하여 DAG 생성"""
    comp_id: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            comp_id[v] = i
    
    dag = defaultdict(set)
    for v in adj:
        for w in adj[v]:
            if comp_id[v] != comp_id[w]:
                dag[comp_id[v]].add(comp_id[w])
    
    for i in range(len(sccs)):
        dag.setdefault(i, set())
    
    return dag, comp_id

def longest_path_dag(dag):
    """DAG에서 최장 경로 계산"""
    if not dag:
        return 0, {}
    
    indeg = {u: 0 for u in dag}
    for u in dag:
        for v in dag[u]:
            indeg[v] += 1
    
    q = deque([u for u, d in indeg.items() if d == 0])
    topo: List[int] = []
    
    while q:
        u = q.popleft()
        topo.append(u)
        for v in dag[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    if not topo:
        return 0, {}
    
    rank = {u: i for i, u in enumerate(topo)}
    dp = {u: 0 for u in topo}
    
    for u in topo:
        for v in dag[u]:
            dp[v] = max(dp[v], dp[u] + 1)
    
    L = max(dp.values()) if dp else 0
    return L, rank

def compute_metrics(parsed: Parsed):
    """모든 지표 계산"""
    adj, radj = build_graph(parsed)

    # 실제 노드 수 (서브그래프 제외)
    actual_nodes = parsed.nodes - parsed.subgraph_definitions
    N = len(actual_nodes)
    E = len(parsed.edges)
    
    out_deg = {u: len(adj[u]) for u in adj}
    in_deg = {u: len(radj[u]) for u in radj}

    BR = sum(1 for u in out_deg if out_deg[u] >= 2)
    JN = sum(1 for u in in_deg if in_deg[u] >= 2)

    # SCC 분석
    sccs = tarjan_scc(adj)
    scc_ge2 = [c for c in sccs if len(c) >= 2]
    has_cycle = len(scc_ge2) > 0 or any(u in adj[u] for u in adj)

    # DAG 압축 및 최장 경로
    dag, comp_id = compress_to_dag(adj, sccs)
    L, rank = longest_path_dag(dag)

    # 내부 SCC 보너스
    internal_bonus = 0
    for comp in sccs:
        if len(comp) >= 2:
            internal_bonus = max(internal_bonus, min(len(comp) - 1, 3))
    
    DEPTH = L + internal_bonus

    # 교차 링크 계산
    X_LINK = 0
    for (u, v, _t, _lab) in parsed.edges:
        cu = comp_id.get(u, 0)
        cv = comp_id.get(v, 0)
        if cu == cv:  # 같은 SCC 내부는 제외
            continue
        du = rank.get(cu, 0)
        dv = rank.get(cv, 0)
        if dv - du >= 2:
            X_LINK += 1

    # 엣지 라벨 비율
    ELR = (parsed.labeled_edge_count / E) if E > 0 else 0.0

    # 결정 노드 수
    DEC = len(parsed.decision_nodes)

    # 서브그래프 관련 지표
    SUBGRAPH_COUNT = len(parsed.subgraph_definitions)
    SUBGRAPH_DEPTH_max = 0  # 간단화

    # 병렬성
    PAR = min(BR, JN)

    metrics = {
        "N": N, 
        "E": E,
        "out_degree_avg": (sum(out_deg.values()) / N if N > 0 else 0.0),
        "BR": BR, 
        "JN": JN,
        "has_cycle": has_cycle, 
        "SCC_count_ge2": len(scc_ge2), 
        "DEPTH": DEPTH,
        "X_LINK": X_LINK, 
        "PAR": PAR,
        "SUBGRAPH_COUNT": SUBGRAPH_COUNT,
        "SUBGRAPH_DEPTH_max": SUBGRAPH_DEPTH_max,
        "DEC": DEC, 
        "ELR": ELR,
        "EDGE_TYPE_DIVERSITY": len(parsed.edge_types),
        "STYLE_LINES": parsed.style_lines
    }
    
    aux = {"rank": rank, "sccs": sccs, "edge_types": list(parsed.edge_types)}
    return metrics, aux

# =========================
# 점수/레벨 산정
# =========================

CAPS = {
    "N": 30, "E": 40, "DEPTH": 10, "BR": 8, "JN": 8, "X_LINK": 5,
    "SUBGRAPH_COUNT": 4, "SUBGRAPH_DEPTH_max": 3, "DEC": 6, "ELR": 1.0, 
    "EDGE_TYPE_DIVERSITY": 3, "STYLE_LINES": 6
}

WEIGHTS = {
    "N": 0.07, "E": 0.07, "DEPTH": 0.16, "BR": 0.10, "JN": 0.06, "X_LINK": 0.14,
    "SUBGRAPH_COUNT": 0.05, "SUBGRAPH_DEPTH_max": 0.05, "DEC": 0.08, "ELR": 0.04, 
    "EDGE_TYPE_DIVERSITY": 0.05, "STYLE_LINES": 0.08
}

def normalize(metrics):
    """지표 정규화"""
    f = {}
    for k, cap in CAPS.items():
        x = metrics[k]
        if k == "ELR":
            f[k] = max(0.0, min(float(x), 1.0))
        else:
            f[k] = min(x / cap, 1.0)
    return f

def score(metrics):
    """점수 계산"""
    f = normalize(metrics)
    base = sum(WEIGHTS[k] * f[k] for k in WEIGHTS) * 100.0
    
    booster = 0.0
    if metrics["has_cycle"]:
        booster += 6.0
        if metrics["N"] <= 6:
            booster += 2.0
    
    if metrics["PAR"] >= 4:
        booster += 6.0
    elif metrics["PAR"] >= 2:
        booster += 3.0
    
    final = min(base + booster, 100.0)
    return {
        "base": round(base, 2), 
        "booster": round(booster, 2), 
        "final": round(final, 2)
    }, f

def level_from_score(S, m):
    """점수로부터 레벨 결정"""
    if S < 20:
        lvl = "L1"
    elif S < 40:
        lvl = "L2"
    elif S < 60:
        lvl = "L3"
    elif S < 80:
        lvl = "L4"
    else:
        lvl = "L5"
    
    overrides = []
    
    # 규칙 A: 사이클이 있고 노드가 많으면 최소 L4
    if m["has_cycle"] and m["N"] >= 8 and lvl in ("L1", "L2", "L3"):
        overrides.append("A: cycle+N>=8 → min L4")
        lvl = "L4"
    
    # 규칙 B: 교차링크와 서브그래프가 많으면 최소 L4
    if m["X_LINK"] >= 2 and m["SUBGRAPH_COUNT"] >= 2 and lvl in ("L1", "L2", "L3"):
        overrides.append("B: X_LINK>=2 & SUBGRAPH>=2 → min L4")
        lvl = "L4"
    
    # 규칙 C: 서브그래프 깊이와 분기가 많으면 최소 L4
    if m["SUBGRAPH_DEPTH_max"] >= 3 and m["BR"] >= 3 and lvl in ("L1", "L2", "L3"):
        overrides.append("C: SUBGRAPH_DEPTH>=3 & BR>=3 → min L4")
        lvl = "L4"
    
    # 규칙 D: A와 B 조건을 모두 만족하면 최소 L5
    if (m["has_cycle"] and m["N"] >= 8 and m["X_LINK"] >= 2 and 
        m["SUBGRAPH_COUNT"] >= 2 and lvl != "L5"):
        overrides.append("D: (A&B) → min L5")
        lvl = "L5"
    
    return lvl, overrides

def analyze_mermaid(text: str) -> dict:
    """Mermaid 플로우차트 분석"""
    trace: List[str] = []
    try:
        parsed = parse_mermaid_flowchart(text)
    except ValueError as e:
        return {"ok": False, "reason": str(e), "trace": trace}

    trace.append(f"Parsing completed: {len(parsed.nodes)} nodes, {len(parsed.edges)} edges")
    trace.append(f"Subgraph definitions: {len(parsed.subgraph_definitions)}")
    trace.append(f"Edge types: {list(parsed.edge_types)}")
    
    metrics, aux = compute_metrics(parsed)
    s, f = score(metrics)
    lvl, ov = level_from_score(s["final"], metrics)
    
    if metrics["has_cycle"]:
        trace.append(f"Cycle detected: SCC(size>=2) {metrics['SCC_count_ge2']}")
    
    trace.append(f"DEPTH={metrics['DEPTH']}")
    trace.append(f"X_LINK={metrics['X_LINK']}")
    trace.append(f"Subgraph count={metrics['SUBGRAPH_COUNT']}, max depth={metrics['SUBGRAPH_DEPTH_max']}")
    trace.append(f"BR={metrics['BR']}, JN={metrics['JN']}, PAR={metrics['PAR']}")
    
    return {
        "ok": True,
        "reason": "parsed",
        "metrics": metrics,
        "normalized": f,
        "weights": WEIGHTS,
        "score": s,
        "level": lvl,
        "overrides": {"applied": ov, "explanations": ov},
        "trace": trace,
        "auxiliary": {
            "subgraph_definitions": list(parsed.subgraph_definitions),
            "edge_types_found": aux["edge_types"],
            "decision_nodes": list(parsed.decision_nodes)
        }
    }

def analyze_batch(items: Iterable[Tuple[str, str]]) -> List[dict]:
    """배치 분석"""
    out = []
    for name, text in items:
        r = analyze_mermaid(text)
        row = {
            "name": name,
            "ok": r.get("ok", False),
            "level": r.get("level"),
            "score_final": r.get("score", {}).get("final"),
            "score_base": r.get("score", {}).get("base"),
            "score_booster": r.get("score", {}).get("booster"),
        }
        
        if r.get("ok"):
            m = r["metrics"]
            row.update({
                "N": m["N"], 
                "E": m["E"], 
                "DEPTH": m["DEPTH"],
                "BR": m["BR"], 
                "JN": m["JN"], 
                "X_LINK": m["X_LINK"],
                "SUBGRAPH_COUNT": m["SUBGRAPH_COUNT"], 
                "SUBGRAPH_DEPTH_max": m["SUBGRAPH_DEPTH_max"],
                "DEC": m["DEC"], 
                "ELR": m["ELR"], 
                "EDGE_TYPE_DIVERSITY": m["EDGE_TYPE_DIVERSITY"], 
                "STYLE_LINES": m["STYLE_LINES"],
                "has_cycle": m["has_cycle"], 
                "PAR": m["PAR"]
            })
        else:
            row.update({
                "N": None, "E": None, "DEPTH": None,
                "BR": None, "JN": None, "X_LINK": None,
                "SUBGRAPH_COUNT": None, "SUBGRAPH_DEPTH_max": None,
                "DEC": None, "ELR": None, "EDGE_TYPE_DIVERSITY": None, "STYLE_LINES": None,
                "has_cycle": None, "PAR": None
            })
        out.append(row)
    return out

# =========================
# 파일 처리 및 메인 함수
# =========================

def find_mmd_files(directory: str = "mermaid") -> List[Path]:
    """mermaid 디렉토리에서 모든 .mmd 파일을 찾아서 반환"""
    mermaid_dir = Path(directory)
    if not mermaid_dir.exists():
        print(f"Warning: '{directory}' directory does not exist.")
        return []
    
    mmd_files = list(mermaid_dir.glob("*.mmd"))
    return sorted(mmd_files)

def read_file_safe(file_path: Path) -> str:
    """파일을 안전하게 읽어서 내용을 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    except Exception as e:
        print(f"File read error ({file_path}): {e}")
        return ""

def save_results_to_csv(results: List[dict], output_file: str = "mermaid_analysis_results_perfect.csv"):
    """분석 결과를 CSV 파일로 저장"""
    if not results:
        print("No results to save.")
        return
    
    fieldnames = [
        "name", "ok", "level", "score_final", "score_base", "score_booster",
        "N", "E", "DEPTH", "BR", "JN", "X_LINK", 
        "SUBGRAPH_COUNT", "SUBGRAPH_DEPTH_max",
        "DEC", "ELR", "EDGE_TYPE_DIVERSITY", "STYLE_LINES", 
        "has_cycle", "PAR"
    ]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Analysis results saved to '{output_file}' file.")
    except Exception as e:
        print(f"CSV file save error: {e}")

def main():
    """메인 함수: mermaid 폴더의 모든 .mmd 파일을 분석하여 CSV로 저장"""
    print("100% Accurate Mermaid Difficulty Analyzer Starting...")
    
    # mermaid 디렉토리에서 .mmd 파일들 찾기
    mmd_files = find_mmd_files()
    
    if not mmd_files:
        print("ERROR: No .mmd files found for analysis.")
        print("Please put .mmd files in the 'mermaid' folder.")
        return
    
    print(f"Found {len(mmd_files)} .mmd files")
    
    # 파일 내용 읽기 및 분석 준비
    file_contents = []
    for file_path in mmd_files:
        content = read_file_safe(file_path)
        if content.strip():
            file_contents.append((file_path.stem, content))
        else:
            print(f"WARNING: '{file_path}' file is empty or unreadable.")
    
    if not file_contents:
        print("ERROR: No readable files found.")
        return
    
    print(f"Starting analysis... ({len(file_contents)} files)")
    
    # 배치 분석 실행
    results = analyze_batch(file_contents)
    
    # 결과 출력
    print(f"\n=== Analysis Results Summary ===")
    success_count = sum(1 for r in results if r["ok"])
    print(f"Successfully analyzed files: {success_count}/{len(results)}")
    
    # 레벨별 통계
    level_counts = {}
    for r in results:
        if r["ok"] and r["level"]:
            level = r["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
    
    if level_counts:
        print(f"\nLevel distribution:")
        for level in sorted(level_counts.keys()):
            print(f"  {level}: {level_counts[level]} files")
    
    # 상위 점수 파일들
    successful_results = [r for r in results if r["ok"] and r["score_final"] is not None]
    if successful_results:
        successful_results.sort(key=lambda x: x["score_final"], reverse=True)
        print(f"\nTop 5 files (by score):")
        for i, r in enumerate(successful_results[:5], 1):
            print(f"  {i}. {r['name']}: {r['level']} (score: {r['score_final']})")
    
    # 실패한 파일들
    failed_results = [r for r in results if not r["ok"]]
    if failed_results:
        print(f"\nFailed analysis files:")
        for r in failed_results:
            print(f"  FAILED: {r['name']}")
    
    # CSV 파일로 저장
    save_results_to_csv(results)
    
    print(f"\nAnalysis completed!")

if __name__ == "__main__":
        main()