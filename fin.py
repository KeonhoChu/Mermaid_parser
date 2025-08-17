# Mermaid Flowchart 난이도 산정기 (최종본)
# - 라벨 엣지, 점선/물결/이중화살표, & 다중 연결, subgraph, 루프/SCC, 교차링크 지원
# - 외부 패키지 불필요 (표준 라이브러리만 사용)
# - mermaid 폴더의 모든 .mmd 파일을 분석하여 CSV로 저장

import re
import os
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Iterable
from pathlib import Path

# =========================
# 1) 정규식/토큰 정의
# =========================
ID_RE = r"[A-Za-z0-9_\.\$\-]+"     # 노드 ID (라벨은 자유롭게 장식부에 들어감)
NODE_DECOR = r"(?:\(\([^\)]*\)\)|\([^\)]*\)|\[\[[^\]]*\]\]|\[[^\]]*\]|\{[^\}]*\}|>>[^>]*>>)?"
NODE_TOKEN = rf"({ID_RE})({NODE_DECOR})"

# (A) 라벨 포함 엣지:  A -- label --> B
EDGE_LABELED_RE = re.compile(
    rf"^\s*{NODE_TOKEN}\s*--\s*([^-]+?)\s*-->\s*{NODE_TOKEN}\s*$"
)
# 그룹: 1 src_id, 2 src_decor, 3 edge_label, 4 dst_id, 5 dst_decor

# (B) 라벨 없는 엣지:  A --> B,  A -.-> B,  A ==> B,  A ~~~> B 등
# 수정: 더 간단한 화살표 패턴으로 변경
EDGE_UNLABELED_RE = re.compile(
    rf"^\s*{NODE_TOKEN}\s*(-->|\.->|\.-\.|==>|~~~>|--)\s*{NODE_TOKEN}\s*$"
)
# 그룹: 1 src_id, 2 src_decor, 3 arrow, 4 dst_id, 5 dst_decor

# --- & 다중 연결 확장을 위한 그룹 패턴 ---
# 더 간단하고 강건한 패턴으로 수정
NODE_GROUP = rf"[A-Za-z0-9_\.\$\-]+(?:\s*&\s*[A-Za-z0-9_\.\$\-]+)*"

EDGE_LABELED_GROUP_RE = re.compile(
    rf"^\s*({NODE_GROUP})\s*--\s*([^-]+?)\s*-->\s*({NODE_GROUP})\s*$"
)
EDGE_UNLABELED_GROUP_RE = re.compile(
    rf"^\s*({NODE_GROUP})\s*(-->|\.->|\.-\.|==>|~~~>|--)\s*({NODE_GROUP})\s*$"
)

# --- 기타 문법 ---
FLOWCHART_HEAD_RE = re.compile(r"^\s*flowchart\b", re.IGNORECASE)
SUBGRAPH_START_RE = re.compile(rf"^\s*subgraph\s+({ID_RE})(?:\s*\[.*?\])?\s*$", re.IGNORECASE)
SUBGRAPH_END_RE   = re.compile(r"^\s*end\s*$", re.IGNORECASE)
COMMENT_RE        = re.compile(r"^\s*%%")
STYLE_LIKE_RE     = re.compile(r"^\s*(style|classDef|link|click)\b")

# 단독 노드 선언 패턴들 (누락된 부분 추가)
NODE_DECL_RES = [
    re.compile(rf"^\s*({ID_RE})\s*\[\[(.*?)\]\]\s*$"),     # [[label]]
    re.compile(rf"^\s*({ID_RE})\s*\[(.*?)\]\s*$"),        # [label]
    re.compile(rf"^\s*({ID_RE})\s*\(\((.*?)\)\)\s*$"),    # ((label))
    re.compile(rf"^\s*({ID_RE})\s*\((.*?)\)\s*$"),        # (label)
    re.compile(rf"^\s*({ID_RE})\s*\{{(.*?)\}}\s*$"),      # {label}
    re.compile(rf"^\s*({ID_RE})\s*>>(.*?)>>\s*$"),        # >>label>>
]

# =========================
# 2) 자료구조
# =========================
@dataclass
class Parsed:
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str, str, str]] = field(default_factory=set)  # (src,dst,type,label)
    node_labels: Dict[str, str] = field(default_factory=dict)           # id -> label
    node_shapes: Dict[str, Dict[str, str]] = field(default_factory=dict)# id -> {'shape': ...}
    node_subgraphs: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    subgraph_ids: Set[str] = field(default_factory=set)
    style_lines: int = 0
    decision_nodes: Set[str] = field(default_factory=set)               # {} or label '?'
    edge_types: Set[str] = field(default_factory=set)                   # '-->', '-.->', '==>', ...
    labeled_edge_count: int = 0

# =========================
# 3) 유틸: 장식 해석/노드 등록
# =========================
def _shape_and_label_from_decor(decor: str):
    if not decor:
        return "rect", ""
    s = decor.strip()
    if s.startswith(">>") and s.endswith(">>"):
        return "subroutine", s[2:-2]
    if s.startswith("((") and s.endswith("))"):
        return "double_circle", s[2:-2]
    if s.startswith("(") and s.endswith(")"):
        return "circle", s[1:-1]
    if s.startswith("[[") and s.endswith("]]"):
        return "double_rect", s[2:-2]
    if s.startswith("[") and s.endswith("]"):
        return "rect", s[1:-1]
    if s.startswith("{") and s.endswith("}"):
        return "diamond", s[1:-1]
    return "rect", s

def _register_node_inline(p: Parsed, nid: str, decor: str, sg_stack: List[str]):
    if nid not in p.nodes:
        p.nodes.add(nid)
    shape, label = _shape_and_label_from_decor(decor or "")
    if nid not in p.node_labels:
        p.node_labels[nid] = label
    if nid not in p.node_shapes:
        p.node_shapes[nid] = {"shape": shape}
    if shape == "diamond" or "?" in label:
        p.decision_nodes.add(nid)
    for sg in sg_stack:
        p.node_subgraphs[nid].append(sg)

# =========================
# 4) & 다중 연결 전개 전처리
# =========================
def _split_node_group_to_pairs(group_text: str):
    """'N1 & N2 & N3' → [(N1, ""), (N2, ""), (N3, "")] 형태로 분해"""
    parts = re.split(r"\s*&\s*", group_text.strip())
    out = []
    for p in parts:
        # 우선 간단한 ID만 처리 (장식은 나중에 처리)
        p = p.strip()
        if re.match(rf"^{ID_RE}$", p):
            out.append((p, ""))
        else:
            # 장식이 있는 경우 처리
            m = re.match(rf"^\s*({ID_RE})({NODE_DECOR})\s*$", p)
            if m:
                out.append((m.group(1), m.group(2)))
            else:
                # 매칭되지 않는 경우도 ID로 처리
                out.append((p, ""))
    return out

def expand_ampersand_edges(raw_line: str) -> List[str]:
    """& 축약을 모든 조합의 단일 엣지 라인으로 전개"""
    # 먼저 & 패턴이 있는지 확인
    if "&" not in raw_line:
        return [raw_line]  # & 패턴이 없으면 원본 그대로 반환
    
    m = EDGE_LABELED_GROUP_RE.match(raw_line)
    if m:
        left_group, label, right_group = m.group(1), (m.group(2) or "").strip(), m.group(3)
        L = _split_node_group_to_pairs(left_group)
        R = _split_node_group_to_pairs(right_group)
        result = [f"{lid}{ldecor} -- {label} --> {rid}{rdecor}"
                for (lid, ldecor) in L for (rid, rdecor) in R]
        return result

    m = EDGE_UNLABELED_GROUP_RE.match(raw_line)
    if m:
        left_group, arrow, right_group = m.group(1), m.group(2), m.group(3)
        L = _split_node_group_to_pairs(left_group)
        R = _split_node_group_to_pairs(right_group)
        result = [f"{lid}{ldecor} {arrow} {rid}{rdecor}"
                for (lid, ldecor) in L for (rid, rdecor) in R]
        return result

    return [raw_line]  # 전개 불필요

# =========================
# 5) 파서
# =========================
def parse_mermaid_flowchart(text: str) -> Parsed:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not COMMENT_RE.match(ln)]
    if not lines or not FLOWCHART_HEAD_RE.match(lines[0]):
        raise ValueError("Not a flowchart or empty")

    p = Parsed()
    sg_stack: List[str] = []
    
    # 헤더 다음 라인부터 처리
    for raw in lines[1:]:
        # 먼저 & 축약을 전개하여, 여러 단일 엣지 라인으로 변환
        expanded_lines = expand_ampersand_edges(raw)
        
        for ln in expanded_lines:
            # subgraph 블록
            m_sg = SUBGRAPH_START_RE.match(ln)
            if m_sg:
                sg_id = m_sg.group(1)
                sg_stack.append(sg_id)
                p.subgraph_ids.add(sg_id)
                continue
            if SUBGRAPH_END_RE.match(ln):
                if sg_stack:
                    sg_stack.pop()
                continue

            # 스타일/클래스/링크/클릭 → 카운트만
            if STYLE_LIKE_RE.match(ln):
                p.style_lines += 1
                continue

            # (A) 라벨 포함 엣지
            m_lab = EDGE_LABELED_RE.match(ln)
            if m_lab:
                src_id, src_decor = m_lab.group(1), m_lab.group(2)
                edge_label        = (m_lab.group(3) or "").strip()
                dst_id, dst_decor = m_lab.group(4), m_lab.group(5)
                _register_node_inline(p, src_id, src_decor, sg_stack)
                _register_node_inline(p, dst_id, dst_decor, sg_stack)
                if edge_label:
                    p.labeled_edge_count += 1
                edge_type = "-->"  # 라벨 엣지는 항상 --> 로 끝남
                p.edge_types.add(edge_type)
                p.edges.add((src_id, dst_id, edge_type, edge_label))
                continue

            # (B) 라벨 없는 엣지
            m_un = EDGE_UNLABELED_RE.match(ln)
            if m_un:
                src_id, src_decor = m_un.group(1), m_un.group(2)
                arrow             = m_un.group(3)
                dst_id, dst_decor = m_un.group(4), m_un.group(5)
                _register_node_inline(p, src_id, src_decor, sg_stack)
                _register_node_inline(p, dst_id, dst_decor, sg_stack)
                edge_type = "-->" if arrow.startswith("--") else arrow
                p.edge_types.add(edge_type)
                p.edges.add((src_id, dst_id, edge_type, ""))  # 라벨 없음
                continue

            # 단독 노드 선언
            matched_decl = False
            for rx in NODE_DECL_RES:
                m = rx.match(ln)
                if m:
                    nid, label = m.group(1), m.group(2)
                    p.nodes.add(nid)
                    # 모양 판정
                    shape = "rect"
                    if rx.pattern.endswith(r"\}\s*$"): 
                        shape = "diamond"
                        p.decision_nodes.add(nid)
                    elif ">>" in rx.pattern: 
                        shape = "subroutine"
                    elif r"\(\(" in rx.pattern: 
                        shape = "double_circle"
                    elif r"\(" in rx.pattern: 
                        shape = "circle"
                    elif r"\[\[" in rx.pattern: 
                        shape = "double_rect"
                    
                    p.node_shapes[nid] = {"shape": shape}
                    p.node_labels[nid] = str(label)
                    if "?" in str(label):
                        p.decision_nodes.add(nid)
                    for sg in sg_stack:
                        p.node_subgraphs[nid].append(sg)
                    matched_decl = True
                    break
            if matched_decl:
                continue

            # 아이디 단독 라인
            tid = re.match(rf"^\s*({ID_RE})\s*$", ln)
            if tid:
                nid = tid.group(1)
                p.nodes.add(nid)
                if nid not in p.node_labels:
                    p.node_labels[nid] = ""
                if nid not in p.node_shapes:
                    p.node_shapes[nid] = {"shape": "rect"}
                for sg in sg_stack:
                    p.node_subgraphs[nid].append(sg)
                continue

            # 그 외 문법은 난이도 산정에서 무시
    return p

# =========================
# 6) 그래프/지표 계산
# =========================
def build_graph(parsed: Parsed):
    adj = defaultdict(set)
    radj = defaultdict(set)
    for (u, v, _t, _lab) in parsed.edges:
        adj[u].add(v)
        radj[v].add(u)
    for n in parsed.nodes:
        adj.setdefault(n, set())
        radj.setdefault(n, set())
    return adj, radj

def tarjan_scc(adj: Dict[str, Set[str]]):
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        idx[v] = index; low[v] = index; index += 1
        stack.append(v); onstack.add(v)
        for w in adj[v]:
            if w not in idx:
                strongconnect(w); low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop(); onstack.discard(w)
                comp.append(w)
                if w == v: break
            sccs.append(comp)

    for v in list(adj.keys()):
        if v not in idx:
            strongconnect(v)
    return sccs

def compress_to_dag(adj, sccs):
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
    indeg = {u:0 for u in dag}
    for u in dag:
        for v in dag[u]:
            indeg[v] += 1
    q = deque([u for u,d in indeg.items() if d==0])
    topo: List[int] = []
    while q:
        u = q.popleft(); topo.append(u)
        for v in dag[u]:
            indeg[v] -= 1
            if indeg[v]==0: q.append(v)
    if not topo:  # 빈 그래프 처리
        return 0, {}
    rank = {u:i for i,u in enumerate(topo)}
    dp = {u:0 for u in topo}
    for u in topo:
        for v in dag[u]:
            dp[v] = max(dp[v], dp[u]+1)
    L = max(dp.values()) if dp else 0
    return L, rank

def compute_metrics(parsed: Parsed):
    adj, radj = build_graph(parsed)

    N = len(parsed.nodes)  # 수정: adj 대신 parsed.nodes 사용
    E = len(parsed.edges)  # 수정: 실제 엣지 개수 사용
    out_deg = {u: len(adj[u]) for u in adj}
    in_deg  = {u: len(radj[u]) for u in radj}

    BR = sum(1 for u in out_deg if out_deg[u] >= 2)
    JN = sum(1 for u in in_deg  if in_deg[u]  >= 2)

    # SCCs / cycle
    sccs = tarjan_scc(adj)
    scc_ge2 = [c for c in sccs if len(c) >= 2]
    has_cycle = len(scc_ge2) > 0 or any(u in adj[u] for u in adj)

    # Compress + longest path
    dag, comp_id = compress_to_dag(adj, sccs)
    L, rank = longest_path_dag(dag)

    # 내부 SCC 보너스(깊이 보정)
    internal_bonus = 0
    for comp in sccs:
        if len(comp) >= 2:
            internal_bonus = max(internal_bonus, min(len(comp)-1, 3))
    DEPTH = L + internal_bonus

    # 교차 링크 (Δ>=2 on DAG ranks)
    X_LINK = 0
    for (u, v, _t, _lab) in parsed.edges:
        cu, cv = comp_id.get(u, 0), comp_id.get(v, 0)  # get 사용하여 안전하게 처리
        if cu == cv:  # 같은 SCC 내부 엣지는 제외
            continue
        du = rank.get(cu, 0); dv = rank.get(cv, 0)
        if dv - du >= 2:
            X_LINK += 1

    # Edge labels ratio
    ELR = (parsed.labeled_edge_count / E) if E > 0 else 0.0

    # Decisions: 다이아몬드 + 라벨 '?' 포함
    DEC = len(parsed.decision_nodes)
    for nid, lbl in parsed.node_labels.items():
        if "?" in str(lbl) and nid not in parsed.decision_nodes:
            DEC += 1

    # Subgraphs
    SUB = len(parsed.subgraph_ids)
    SUB_DEPTH_max = 0
    for nid, sgs in parsed.node_subgraphs.items():
        SUB_DEPTH_max = max(SUB_DEPTH_max, len(sgs))

    # 병렬성 근사
    PAR = min(BR, JN)

    metrics = {
        "N": N, "E": E,
        "out_degree_avg": (sum(out_deg.values())/N if N>0 else 0.0),
        "BR": BR, "JN": JN,
        "has_cycle": has_cycle, "SCC_count_ge2": len(scc_ge2), "DEPTH": DEPTH,
        "X_LINK": X_LINK, "PAR": PAR,
        "SUB": SUB, "SUB_DEPTH_max": SUB_DEPTH_max,
        "DEC": DEC, "ELR": ELR,
        "ETD": len(parsed.edge_types), "STY": parsed.style_lines
    }
    aux = {"rank": rank, "sccs": sccs}
    return metrics, aux

# =========================
# 7) 점수/레벨 산정
# =========================
CAPS = {
    "N":30, "E":40, "DEPTH":10, "BR":8, "JN":8, "X_LINK":5,
    "SUB":4, "SUB_DEPTH_max":3, "DEC":6, "ELR":1.0, "ETD":3, "STY":6
}
WEIGHTS = {
    "N":0.07, "E":0.07, "DEPTH":0.16, "BR":0.10, "JN":0.06, "X_LINK":0.14,
    "SUB":0.05, "SUB_DEPTH_max":0.05, "DEC":0.08, "ELR":0.04, "ETD":0.05, "STY":0.08
}

def normalize(metrics):
    f = {}
    for k, cap in CAPS.items():
        x = metrics[k]
        if k == "ELR":
            f[k] = max(0.0, min(float(x), 1.0))
        else:
            f[k] = min(x/cap, 1.0)
    return f

def score(metrics):
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
    return {"base": round(base,2), "booster": round(booster,2), "final": round(final,2)}, f

def level_from_score(S, m):
    lvl = "L1" if S < 20 else "L2" if S < 40 else "L3" if S < 60 else "L4" if S < 80 else "L5"
    overrides = []
    if m["has_cycle"] and m["N"] >= 8 and lvl in ("L1","L2","L3"):
        overrides.append("A: cycle+N>=8 → min L4"); lvl = "L4"
    if m["X_LINK"] >= 2 and m["SUB"] >= 2 and lvl in ("L1","L2","L3"):
        overrides.append("B: X_LINK>=2 & SUB>=2 → min L4"); lvl = "L4"
    if m["SUB_DEPTH_max"] >= 3 and m["BR"] >= 3 and lvl in ("L1","L2","L3"):
        overrides.append("C: SUB_DEPTH>=3 & BR>=3 → min L4"); lvl = "L4"
    if m["has_cycle"] and m["N"] >= 8 and m["X_LINK"] >= 2 and m["SUB"] >= 2 and lvl != "L5":
        overrides.append("D: (A&B) → min L5"); lvl = "L5"
    return lvl, overrides

# =========================
# 8) 분석 API
# =========================
def analyze_mermaid(text: str) -> dict:
    trace: List[str] = []
    try:
        parsed = parse_mermaid_flowchart(text)
    except ValueError as e:
        return {"ok": False, "reason": str(e), "trace": trace}

    trace.append(f"Parsed flowchart; nodes so far {len(parsed.nodes)}, edges so far {len(parsed.edges)}")
    metrics, _ = compute_metrics(parsed)
    s, f = score(metrics)
    lvl, ov = level_from_score(s["final"], metrics)
    if metrics["has_cycle"]:
        trace.append(f"SCCs with size>=2: {metrics['SCC_count_ge2']}")
    trace.append(f"DEPTH={metrics['DEPTH']}")
    trace.append(f"X_LINK={metrics['X_LINK']}; SUB={metrics['SUB']}; SUB_DEPTH_max={metrics['SUB_DEPTH_max']}")
    trace.append(f"BR={metrics['BR']}; JN={metrics['JN']}; PAR={metrics['PAR']}")
    return {
        "ok": True,
        "reason": "parsed",
        "metrics": metrics,
        "normalized": f,
        "weights": WEIGHTS,
        "score": s,
        "level": lvl,
        "overrides": {"applied": ov, "explanations": ov},
        "trace": trace
    }

def analyze_batch(items: Iterable[Tuple[str, str]]) -> List[dict]:
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
                "N": m["N"], "E": m["E"], "DEPTH": m["DEPTH"],
                "BR": m["BR"], "JN": m["JN"], "X_LINK": m["X_LINK"],
                "SUB": m["SUB"], "SUB_DEPTH_max": m["SUB_DEPTH_max"],
                "DEC": m["DEC"], "ELR": m["ELR"], "ETD": m["ETD"], "STY": m["STY"],
                "has_cycle": m["has_cycle"], "PAR": m["PAR"]
            })
        else:
            row.update({
                "N": None, "E": None, "DEPTH": None,
                "BR": None, "JN": None, "X_LINK": None,
                "SUB": None, "SUB_DEPTH_max": None,
                "DEC": None, "ELR": None, "ETD": None, "STY": None,
                "has_cycle": None, "PAR": None
            })
        out.append(row)
    return out

# =========================
# 9) 파일 처리 및 메인 함수
# =========================
def find_mmd_files(directory: str = "mermaid") -> List[Path]:
    """mermaid 디렉토리에서 모든 .mmd 파일을 찾아서 반환"""
    mermaid_dir = Path(directory)
    if not mermaid_dir.exists():
        print(f"경고: '{directory}' 디렉토리가 존재하지 않습니다.")
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
        print(f"파일 읽기 오류 ({file_path}): {e}")
        return ""

def save_results_to_csv(results: List[dict], output_file: str = "mermaid_analysis_results.csv"):
    """분석 결과를 CSV 파일로 저장"""
    if not results:
        print("저장할 결과가 없습니다.")
        return
    
    # CSV 헤더 정의
    fieldnames = [
        "name", "ok", "level", "score_final", "score_base", "score_booster",
        "N", "E", "DEPTH", "BR", "JN", "X_LINK", "SUB", "SUB_DEPTH_max",
        "DEC", "ELR", "ETD", "STY", "has_cycle", "PAR"
    ]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"분석 결과가 '{output_file}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 오류: {e}")

def main():
    """메인 함수: mermaid 폴더의 모든 .mmd 파일을 분석하여 CSV로 저장"""
    print("Mermaid 난이도 산정기 시작...")
    
    # mermaid 디렉토리에서 .mmd 파일들 찾기
    mmd_files = find_mmd_files()
    
    if not mmd_files:
        print("분석할 .mmd 파일이 없습니다.")
        print("'mermaid' 폴더에 .mmd 파일들을 넣어주세요.")
        return
    
    print(f"발견된 .mmd 파일: {len(mmd_files)}개")
    
    # 파일 내용 읽기 및 분석 준비
    file_contents = []
    for file_path in mmd_files:
        content = read_file_safe(file_path)
        if content.strip():  # 빈 파일이 아닌 경우만 추가
            file_contents.append((file_path.stem, content))  # 파일명(확장자 제외)과 내용
        else:
            print(f"경고: '{file_path}' 파일이 비어있거나 읽을 수 없습니다.")
    
    if not file_contents:
        print("읽을 수 있는 파일이 없습니다.")
        return
    
    print(f"분석 시작... ({len(file_contents)}개 파일)")
    
    # 배치 분석 실행
    results = analyze_batch(file_contents)
    
    # 결과 출력
    print("\n=== 분석 결과 요약 ===")
    success_count = sum(1 for r in results if r["ok"])
    print(f"성공적으로 분석된 파일: {success_count}/{len(results)}")
    
    # 레벨별 통계
    level_counts = {}
    for r in results:
        if r["ok"] and r["level"]:
            level = r["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
    
    if level_counts:
        print("\n레벨별 분포:")
        for level in sorted(level_counts.keys()):
            print(f"  {level}: {level_counts[level]}개")
    
    # 상위 점수 파일들
    successful_results = [r for r in results if r["ok"] and r["score_final"] is not None]
    if successful_results:
        successful_results.sort(key=lambda x: x["score_final"], reverse=True)
        print(f"\n상위 5개 파일 (점수순):")
        for i, r in enumerate(successful_results[:5], 1):
            print(f"  {i}. {r['name']}: {r['level']} (점수: {r['score_final']})")
    
    # 실패한 파일들
    failed_results = [r for r in results if not r["ok"]]
    if failed_results:
        print(f"\n분석 실패한 파일들:")
        for r in failed_results:
            print(f"  - {r['name']}")
    
    # CSV 파일로 저장
    save_results_to_csv(results)
    
    print("\n분석 완료!")


if __name__ == "__main__":
    # 실제 분석 실행
    main()
