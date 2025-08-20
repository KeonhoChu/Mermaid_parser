#!/usr/bin/env python3
"""
100개의 다양한 Mermaid 파일 생성기
level.py의 파싱 정확도를 테스트하기 위한 다양한 패턴 포함
"""

import os
from pathlib import Path

def create_mermaid_files():
    """100개의 다양한 Mermaid 파일 생성"""
    
    mermaid_dir = Path("mermaid")
    mermaid_dir.mkdir(exist_ok=True)
    
    # 기존 test 파일들 정리
    for existing_file in mermaid_dir.glob("test*.mmd"):
        existing_file.unlink()
    
    files = []
    
    # 1. 기본 단순 패턴 (10개)
    for i in range(1, 11):
        content = f"""flowchart TD
    A --> B
    B --> C{i}
"""
        files.append((f"test{i:03d}.mmd", content, 3, 2, 0))
    
    # 2. 서브그래프 기본 패턴 (10개)
    for i in range(11, 21):
        content = f"""flowchart TD
    A --> B
    subgraph S1["Group {i-10}"]
        C --> D
    end
    B --> S1
"""
        files.append((f"test{i:03d}.mmd", content, 4, 3, 1))
    
    # 3. 무방향 엣지 패턴 (10개)
    for i in range(21, 31):
        content = f"""flowchart LR
    A{i-20} --- B{i-20}
    B{i-20} === C{i-20}
    C{i-20} -.- D{i-20}
"""
        files.append((f"test{i:03d}.mmd", content, 4, 3, 0))
    
    # 4. 복잡한 분기 패턴 (10개)
    for i in range(31, 41):
        content = f"""flowchart TD
    A --> B{i-30}
    A --> C{i-30}
    A --> D{i-30}
    B{i-30} --> E{i-30}
    C{i-30} --> E{i-30}
    D{i-30} --> F{i-30}
"""
        files.append((f"test{i:03d}.mmd", content, 6, 6, 0))
    
    # 5. 중첩 서브그래프 패턴 (10개)
    for i in range(41, 51):
        content = f"""flowchart TD
    A --> B
    subgraph S1["Outer {i-40}"]
        subgraph S2["Inner {i-40}"]
            C --> D
        end
        E --> S2
    end
    B --> S1
"""
        files.append((f"test{i:03d}.mmd", content, 5, 4, 2))
    
    # 6. 연쇄 엣지 패턴 (10개)
    for i in range(51, 61):
        content = f"""flowchart LR
    A{i-50} --> B{i-50} --> C{i-50} --> D{i-50} --> E{i-50}
"""
        files.append((f"test{i:03d}.mmd", content, 5, 4, 0))
    
    # 7. & 연산자 패턴 (10개)
    for i in range(61, 71):
        content = f"""flowchart TD
    A{i-60} & B{i-60} --> C{i-60} & D{i-60}
"""
        files.append((f"test{i:03d}.mmd", content, 4, 4, 0))
    
    # 8. 라벨 엣지 패턴 (10개)
    for i in range(71, 81):
        content = f"""flowchart TD
    A --> |label{i-70}| B
    B --> |process{i-70}| C
    C --> |result{i-70}| D
"""
        files.append((f"test{i:03d}.mmd", content, 4, 3, 0))
    
    # 9. 결정 노드 패턴 (10개)
    for i in range(81, 91):
        content = f"""flowchart TD
    A --> B{{Decision {i-80}}}
    B --> |Yes| C
    B --> |No| D
    C --> E
    D --> E
"""
        files.append((f"test{i:03d}.mmd", content, 5, 5, 0))
    
    # 10. 복합 패턴 (10개)
    for i in range(91, 101):
        content = f"""flowchart TD
    Start --> Process{i-90}
    Process{i-90} --> Decision{{Check {i-90}}}
    Decision --> |Pass| subgraph S1["Success Path"]
        Success --> End1
    end
    Decision --> |Fail| subgraph S2["Failure Path"] 
        Retry --> Process{i-90}
        Fail --> End2
    end
    subgraph S3["Log System"]
        Log1 --> Log2
    end
    Success --- Log1
    Fail === Log2
"""
        files.append((f"test{i:03d}.mmd", content, 9, 9, 3))
    
    # 파일 생성 및 검증
    print("100개 Mermaid 테스트 파일 생성 중...")
    
    for filename, content, expected_n, expected_e, expected_s in files:
        filepath = mermaid_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"생성됨: {filename} (예상 N={expected_n}, E={expected_e}, S={expected_s})")
    
    print(f"\n총 {len(files)}개 파일 생성 완료!")
    print("파일들이 mermaid/ 폴더에 저장되었습니다.")
    
    # 예상 결과 요약 저장
    summary_path = "expected_results.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("파일명\t예상_N\t예상_E\t예상_S\n")
        for filename, content, expected_n, expected_e, expected_s in files:
            f.write(f"{filename}\t{expected_n}\t{expected_e}\t{expected_s}\n")
    
    print(f"예상 결과가 {summary_path}에 저장되었습니다.")
    
    return len(files)

if __name__ == "__main__":
    count = create_mermaid_files()
    print(f"\n테스트 준비 완료! {count}개 파일로 level.py 테스트를 시작하세요.")