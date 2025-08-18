import re
import csv
from pathlib import Path
from typing import Set, List, Tuple

def parse_mermaid_accurate(content: str) -> Tuple[int, int, int]:
    """정확한 Mermaid 파서 - 노드, 엣지, 서브그래프 수를 정확히 계산"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    nodes = set()
    edges = []
    subgraphs = set()
    
    for line in lines:
        # 주석과 헤더 건너뛰기
        if line.startswith('%%') or line.lower().startswith('flowchart') or line.lower().startswith('classDef') or line.lower().startswith('class ') or line.lower().startswith('click'):
            continue
            
        # 서브그래프 찾기
        sg_match = re.search(r'subgraph\s+([A-Za-z0-9_]+)', line, re.IGNORECASE)
        if sg_match:
            subgraphs.add(sg_match.group(1))
            continue
            
        # 서브그래프 끝
        if line.lower() == 'end':
            continue
            
        # 모든 노드 ID 추출 (대괄호, 소괄호, 중괄호 안의 내용 포함)
        node_patterns = [
            r'([A-Za-z0-9_]+)\["[^"]*"\]',  # A["label"]
            r'([A-Za-z0-9_]+)\[[^\]]*\]',   # A[label]
            r'([A-Za-z0-9_]+)\([^\)]*\)',   # A(label)  
            r'([A-Za-z0-9_]+)\{[^\}]*\}',   # A{label}
            r'([A-Za-z0-9_]+)\[\[[^\]]*\]\]', # A[[label]]
            r'([A-Za-z0-9_]+)\(\([^\)]*\)\)', # A((label))
        ]
        
        # 모든 노드 패턴에서 노드 추출
        for pattern in node_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if match not in subgraphs:  # 서브그래프는 제외
                    nodes.add(match)
        
        # 화살표 패턴으로 엣지 찾기
        arrow_patterns = [
            r'(\w+)[^-]*?-->.*?(\w+)',      # A --> B (라벨 무시)
            r'(\w+)[^-]*?-\.->.*?(\w+)',    # A -.-> B 
            r'(\w+)[^-]*?==>.*?(\w+)',      # A ==> B
            r'(\w+)[^-]*?<-->.*?(\w+)',     # A <--> B
        ]
        
        for pattern in arrow_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                src, dst = match[0], match[1]
                if src not in subgraphs:
                    nodes.add(src)
                if dst not in subgraphs:
                    nodes.add(dst)
                edges.append((src, dst))
        
        # & 연산자 처리
        if '&' in line and ('-->' in line or '-.->' in line):
            # 간단한 & 확장
            parts = re.split(r'(-\.->|-->)', line)
            if len(parts) >= 3:
                left_part = parts[0].strip()
                right_part = parts[2].strip()
                
                # 양쪽에서 노드 추출
                left_nodes = re.findall(r'([A-Za-z0-9_]+)', left_part)
                right_nodes = re.findall(r'([A-Za-z0-9_]+)', right_part)
                
                # & 기준으로 분리
                if '&' in left_part:
                    left_nodes = [n.strip() for n in left_part.split('&')]
                    left_nodes = [re.findall(r'([A-Za-z0-9_]+)', n)[0] for n in left_nodes if re.findall(r'([A-Za-z0-9_]+)', n)]
                    
                if '&' in right_part:
                    right_nodes = [n.strip() for n in right_part.split('&')]
                    right_nodes = [re.findall(r'([A-Za-z0-9_]+)', n)[0] for n in right_nodes if re.findall(r'([A-Za-z0-9_]+)', n)]
                
                # 모든 조합의 엣지 생성
                for ln in left_nodes:
                    for rn in right_nodes:
                        if ln not in subgraphs:
                            nodes.add(ln)
                        if rn not in subgraphs:
                            nodes.add(rn)
                        edges.append((ln, rn))
    
    # 서브그래프를 노드에서 제거
    actual_nodes = nodes - subgraphs
    
    return len(actual_nodes), len(edges), len(subgraphs)

def test_specific_files():
    """특정 파일들의 정확성 테스트"""
    test_files = ['data01.mmd', 'data02.mmd', 'data10.mmd', 'data04.mmd']
    
    print("Manual verification of parser accuracy:")
    print("=" * 50)
    
    for filename in test_files:
        filepath = Path("mermaid") / filename
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            nodes, edges, subgraphs = parse_mermaid_accurate(content)
            
            print(f"\n{filename}:")
            print(f"  Nodes: {nodes}")
            print(f"  Edges: {edges}") 
            print(f"  Subgraphs: {subgraphs}")
            
            # 내용도 출력
            print("  Content preview:")
            lines = content.strip().split('\n')[:8]
            for line in lines:
                if line.strip():
                    print(f"    {line.strip()}")
            if len(content.strip().split('\n')) > 8:
                print("    ...")

def process_all_files():
    """모든 파일을 간단한 파서로 처리"""
    results = []
    mermaid_dir = Path("mermaid")
    
    for filepath in sorted(mermaid_dir.glob("*.mmd")):
        content = filepath.read_text(encoding='utf-8')
        nodes, edges, subgraphs = parse_mermaid_accurate(content)
        
        results.append({
            'file': filepath.stem,
            'nodes': nodes,
            'edges': edges,
            'subgraphs': subgraphs
        })
        
    # CSV 저장
    csv_path = Path("simple_parser_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'nodes', 'edges', 'subgraphs'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_path}")
    return results

if __name__ == "__main__":
    test_specific_files()
    print("\n" + "=" * 50)
    results = process_all_files()
    
    print(f"\nSummary of {len(results)} files:")
    total_nodes = sum(r['nodes'] for r in results)
    total_edges = sum(r['edges'] for r in results)
    total_subgraphs = sum(r['subgraphs'] for r in results)
    
    print(f"Total nodes: {total_nodes} (avg: {total_nodes/len(results):.1f})")
    print(f"Total edges: {total_edges} (avg: {total_edges/len(results):.1f})")
    print(f"Total subgraphs: {total_subgraphs} (avg: {total_subgraphs/len(results):.1f})")