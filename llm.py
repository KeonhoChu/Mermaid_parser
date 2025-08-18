import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Set, List, Dict, Tuple
from collections import defaultdict

@dataclass
class MermaidResult:
    """간단한 Mermaid 파싱 결과"""
    nodes: Set[str]
    edges: List[Tuple[str, str]]
    subgraphs: Set[str]
    
    def node_count(self) -> int:
        """실제 노드 수 (서브그래프 제외)"""
        return len(self.nodes - self.subgraphs)
    
    def edge_count(self) -> int:
        """엣지 수"""
        return len(self.edges)

def parse_mermaid_simple(content: str) -> MermaidResult:
    """간단한 Mermaid 파서 - 노드와 엣지만 세기"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    nodes = set()
    edges = []
    subgraphs = set()
    in_subgraph = False
    
    # 간단한 ID 패턴 - 알파벳, 숫자, 언더스코어만
    node_pattern = re.compile(r'[A-Za-z0-9_]+')
    
    for line in lines:
        # 주석 건너뛰기
        if line.startswith('%%'):
            continue
            
        # flowchart 선언 건너뛰기
        if line.lower().startswith('flowchart'):
            continue
            
        # 서브그래프 시작
        if line.lower().startswith('subgraph'):
            match = re.search(r'subgraph\s+([A-Za-z0-9_]+)', line, re.IGNORECASE)
            if match:
                subgraph_id = match.group(1)
                subgraphs.add(subgraph_id)
            continue
            
        # 서브그래프 끝
        if line.lower() == 'end':
            continue
            
        # 엣지 찾기 - 다양한 화살표 패턴
        edge_patterns = [
            r'(\w+)\s*-->\s*(\w+)',          # A --> B
            r'(\w+)\s*-\.->\s*(\w+)',        # A -.-> B  
            r'(\w+)\s*==>\s*(\w+)',          # A ==> B
            r'(\w+)\s*<-->\s*(\w+)',         # A <--> B
            r'(\w+)\s*--\s*["\']?[^"\']*["\']?\s*-->\s*(\w+)',  # A -- label --> B
        ]
        
        edge_found = False
        for pattern in edge_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                src, dst = match[0], match[1]
                nodes.add(src)
                nodes.add(dst)
                edges.append((src, dst))
                edge_found = True
                
        if edge_found:
            continue
            
        # & 연산자 처리 (A & B --> C & D)
        if '&' in line and '--' in line:
            # 간단한 & 확장
            parts = line.split('--')
            if len(parts) >= 2:
                left_part = parts[0].strip()
                right_part = parts[-1].strip()
                
                # --> 제거
                right_part = re.sub(r'[->]+', '', right_part).strip()
                
                left_nodes = [n.strip() for n in left_part.split('&')]
                right_nodes = [n.strip() for n in right_part.split('&')]
                
                for ln in left_nodes:
                    ln = re.sub(r'[^\w]', '', ln)
                    if ln:
                        nodes.add(ln)
                        
                for rn in right_nodes:
                    rn = re.sub(r'[^\w]', '', rn)
                    if rn:
                        nodes.add(rn)
                        
                # 모든 조합의 엣지 생성
                for ln in left_nodes:
                    ln = re.sub(r'[^\w]', '', ln)
                    for rn in right_nodes:
                        rn = re.sub(r'[^\w]', '', rn)
                        if ln and rn:
                            edges.append((ln, rn))
            continue
            
        # 단독 노드 선언 찾기
        node_decl_patterns = [
            r'(\w+)\[([^\]]*)\]',      # A[label]
            r'(\w+)\(([^\)]*)\)',      # A(label)
            r'(\w+)\{([^\}]*)\}',      # A{label}
            r'(\w+)\[\[([^\]]*)\]\]',  # A[[label]]
            r'(\w+)\(\(([^\)]*)\)\)',  # A((label))
        ]
        
        for pattern in node_decl_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                node_id = match[0]
                nodes.add(node_id)
                
    return MermaidResult(nodes=nodes, edges=edges, subgraphs=subgraphs)

def process_mermaid_files():
    """mermaid 폴더의 .mmd 파일들을 처리하여 결과를 CSV로 저장"""
    mermaid_dir = Path("mermaid")
    if not mermaid_dir.exists():
        print("ERROR: mermaid folder not found.")
        return
    
    results = []
    mmd_files = list(mermaid_dir.glob("*.mmd"))
    
    print(f"Processing {len(mmd_files)} .mmd files...")
    
    for file_path in sorted(mmd_files):
        print(f"Processing: {file_path.name}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='cp949')
            
        result = parse_mermaid_simple(content)
        
        results.append({
            'file': file_path.stem,
            'nodes': result.node_count(),
            'edges': result.edge_count(),
            'subgraphs': len(result.subgraphs),
            'total_elements': len(result.nodes),
            'node_list': sorted(result.nodes - result.subgraphs),
            'subgraph_list': sorted(result.subgraphs)
        })
        
        print(f"  -> Nodes: {result.node_count()}, Edges: {result.edge_count()}, Subgraphs: {len(result.subgraphs)}")
    
    # CSV 저장
    csv_path = Path("mermaid_analysis_simple.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'nodes', 'edges', 'subgraphs', 'total_elements', 'node_list', 'subgraph_list'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_path}")
    
    # 요약 통계
    total_files = len(results)
    total_nodes = sum(r['nodes'] for r in results)
    total_edges = sum(r['edges'] for r in results)
    total_subgraphs = sum(r['subgraphs'] for r in results)
    
    print(f"\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Total nodes: {total_nodes} (avg: {total_nodes/total_files:.1f})")
    print(f"  Total edges: {total_edges} (avg: {total_edges/total_files:.1f})")
    print(f"  Total subgraphs: {total_subgraphs} (avg: {total_subgraphs/total_files:.1f})")

if __name__ == "__main__":
    process_mermaid_files()