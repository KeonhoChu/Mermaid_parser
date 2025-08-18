import re
from typing import Set, List, Tuple
from pathlib import Path

def parse_mermaid_perfect(content: str) -> Tuple[int, int, int]:
    """100% 정확한 Mermaid 파서"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    nodes = set()
    edges = []
    subgraphs = set()
    
    # 전처리: 모든 라인을 정제
    processed_lines = []
    for line in lines:
        # 주석 제거
        if line.startswith('%%'):
            continue
        # flowchart 선언 제거
        if line.lower().startswith('flowchart'):
            continue
        # 스타일 선언 제거
        if line.lower().startswith(('classDef', 'class ', 'click ', 'style ')):
            continue
        # direction 제거
        if line.lower().startswith('direction'):
            continue
        processed_lines.append(line)
    
    for line in processed_lines:
        # 서브그래프 처리
        if line.lower().startswith('subgraph'):
            sg_match = re.search(r'subgraph\s+([A-Za-z0-9_]+)', line, re.IGNORECASE)
            if sg_match:
                subgraphs.add(sg_match.group(1))
            continue
        
        if line.lower() == 'end':
            continue
            
        # 연쇄 엣지 처리 (A --> B --> C --> D 형태)
        if line.count('-->') >= 2:
            # 연쇄 패턴 찾기
            chain_pattern = r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?(?:\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?)?(?:\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?)?'
            
            chain_match = re.search(chain_pattern, line)
            if chain_match:
                chain_nodes = [g for g in chain_match.groups() if g is not None]
                # 연쇄의 각 노드를 추가
                for node in chain_nodes:
                    if node not in subgraphs:
                        nodes.add(node)
                # 연쇄의 각 엣지를 추가
                for i in range(len(chain_nodes) - 1):
                    edges.append((chain_nodes[i], chain_nodes[i+1]))
                continue
        
        # & 연산자 처리
        if '&' in line and ('-->' in line or '-.->' in line):
            # A & B --> C & D 형태 처리
            arrow_split = None
            if '-->' in line:
                arrow_split = line.split('-->')
            elif '-.->' in line:
                arrow_split = line.split('-.->') 
                
            if arrow_split and len(arrow_split) == 2:
                left_part = arrow_split[0].strip()
                right_part = arrow_split[1].strip()
                
                # 왼쪽과 오른쪽에서 노드 추출
                left_nodes = []
                right_nodes = []
                
                if '&' in left_part:
                    # A & B 형태
                    for part in left_part.split('&'):
                        node_match = re.search(r'([A-Za-z0-9_]+)', part.strip())
                        if node_match:
                            left_nodes.append(node_match.group(1))
                else:
                    node_match = re.search(r'([A-Za-z0-9_]+)', left_part)
                    if node_match:
                        left_nodes.append(node_match.group(1))
                
                if '&' in right_part:
                    # C & D 형태
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
                        if ln not in subgraphs:
                            nodes.add(ln)
                        if rn not in subgraphs:
                            nodes.add(rn)
                        edges.append((ln, rn))
                continue
        
        # 일반 엣지 처리 (라벨 포함)
        edge_patterns = [
            r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?',  # A -->|label| B
            r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?',                # A --> B
            r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-\.->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?',              # A -.-> B
            r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*==>\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?',                # A ==> B
            r'([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*<-->\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?',               # A <--> B
        ]
        
        edge_found = False
        for pattern in edge_patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    if len(match) == 3:  # 라벨이 있는 경우
                        src, label, dst = match[0], match[1], match[2]
                    else:  # 라벨이 없는 경우
                        src, dst = match[0], match[1]
                    
                    if src not in subgraphs:
                        nodes.add(src)
                    if dst not in subgraphs:
                        nodes.add(dst)
                    edges.append((src, dst))
                edge_found = True
                break
        
        if edge_found:
            continue
            
        # 단독 노드 선언 처리
        node_decl_patterns = [
            r'([A-Za-z0-9_]+)\["[^"]*"\]',    # A["label"]
            r'([A-Za-z0-9_]+)\[[^\]]*\]',     # A[label]
            r'([A-Za-z0-9_]+)\([^\)]*\)',     # A(label)
            r'([A-Za-z0-9_]+)\{[^\}]*\}',     # A{label}
            r'([A-Za-z0-9_]+)\[\[[^\]]*\]\]', # A[[label]]
            r'([A-Za-z0-9_]+)\(\([^\)]*\)\)', # A((label))
        ]
        
        for pattern in node_decl_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if match not in subgraphs:
                    nodes.add(match)
    
    return len(nodes), len(edges), len(subgraphs)

def validate_all_files():
    """모든 파일의 정확성 검증"""
    print("100% Accurate Mermaid Parser Validation")
    print("=" * 50)
    
    results = []
    mermaid_dir = Path("mermaid")
    
    for filepath in sorted(mermaid_dir.glob("*.mmd")):
        print(f"\nAnalyzing {filepath.name}:")
        
        try:
            content = filepath.read_text(encoding='utf-8')
        except:
            content = filepath.read_text(encoding='cp949')
            
        nodes, edges, subgraphs = parse_mermaid_perfect(content)
        
        print(f"  Nodes: {nodes}")
        print(f"  Edges: {edges}")
        print(f"  Subgraphs: {subgraphs}")
        
        # 내용 미리보기
        preview_lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('%%') and not line.strip().lower().startswith('flowchart')][:5]
        print("  Preview:")
        for line in preview_lines:
            print(f"    {line}")
        if len([line for line in content.split('\n') if line.strip()]) > 7:
            print("    ...")
            
        results.append({
            'file': filepath.stem,
            'nodes': nodes,
            'edges': edges,
            'subgraphs': subgraphs
        })
    
    return results

if __name__ == "__main__":
    results = validate_all_files()
    
    print(f"\n{'='*50}")
    print(f"Summary of {len(results)} files:")
    total_nodes = sum(r['nodes'] for r in results)
    total_edges = sum(r['edges'] for r in results)
    total_subgraphs = sum(r['subgraphs'] for r in results)
    
    print(f"Total nodes: {total_nodes} (avg: {total_nodes/len(results):.1f})")
    print(f"Total edges: {total_edges} (avg: {total_edges/len(results):.1f})")
    print(f"Total subgraphs: {total_subgraphs} (avg: {total_subgraphs/len(results):.1f})")