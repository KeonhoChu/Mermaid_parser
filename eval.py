#!/usr/bin/env python3
"""
LLM Mermaid Parser 평가 도구
실제 분석 결과와 LLM 예측을 비교하여 성능을 평가합니다.
"""

import os
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class EvalMetrics:
    """평가 지표"""
    accuracy_nodes: float = 0.0
    accuracy_edges: float = 0.0
    accuracy_subgraphs: float = 0.0
    mae_nodes: float = 0.0
    mae_edges: float = 0.0
    mae_subgraphs: float = 0.0
    rmse_nodes: float = 0.0
    rmse_edges: float = 0.0
    rmse_subgraphs: float = 0.0
    success_rate: float = 0.0
    level_accuracy: float = 0.0
    
def load_llm_results(csv_path: str) -> List[Dict[str, Any]]:
    """LLM 결과 CSV 로드"""
    results = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 숫자 필드 변환
                for field in ['N_actual', 'E_actual', 'SUBGRAPH_COUNT_actual', 'n_nodes_pred', 'n_edges_pred', 'n_subgraphs_pred']:
                    if row.get(field) and row[field].strip():
                        try:
                            row[field] = int(row[field])
                        except ValueError:
                            row[field] = 0
                    else:
                        row[field] = 0
                
                # 불린 필드 변환
                row['llm_ok'] = row.get('llm_ok', '').lower() == 'true'
                
                results.append(row)
        print(f"LLM 결과 로드 완료: {len(results)}개 항목")
    except Exception as e:
        print(f"LLM 결과 로드 실패: {e}")
    
    return results

def calculate_accuracy(actual: int, predicted: int, tolerance: int = 0) -> bool:
    """정확도 계산 (허용 오차 포함)"""
    return abs(actual - predicted) <= tolerance

def calculate_mae(actual_list: List[int], predicted_list: List[int]) -> float:
    """평균 절대 오차 (MAE) 계산"""
    if not actual_list or not predicted_list:
        return 0.0
    
    errors = [abs(a - p) for a, p in zip(actual_list, predicted_list)]
    return sum(errors) / len(errors)

def calculate_rmse(actual_list: List[int], predicted_list: List[int]) -> float:
    """평균 제곱근 오차 (RMSE) 계산"""
    if not actual_list or not predicted_list:
        return 0.0
    
    squared_errors = [(a - p) ** 2 for a, p in zip(actual_list, predicted_list)]
    mse = sum(squared_errors) / len(squared_errors)
    return math.sqrt(mse)

def evaluate_performance(results: List[Dict[str, Any]], tolerance: int = 1) -> EvalMetrics:
    """LLM 성능 평가"""
    metrics = EvalMetrics()
    
    if not results:
        return metrics
    
    # 성공한 케이스만 필터링
    successful_results = [r for r in results if r['llm_ok']]
    metrics.success_rate = len(successful_results) / len(results) * 100
    
    if not successful_results:
        print("성공한 LLM 결과가 없습니다.")
        return metrics
    
    # 실제 값과 예측값 리스트
    actual_nodes = [r['N_actual'] for r in successful_results]
    pred_nodes = [r['n_nodes_pred'] for r in successful_results]
    actual_edges = [r['E_actual'] for r in successful_results]
    pred_edges = [r['n_edges_pred'] for r in successful_results]
    actual_subgraphs = [r['SUBGRAPH_COUNT_actual'] for r in successful_results]
    # 서브그래프 예측값 (이미 CSV에서 추출됨)
    pred_subgraphs = [r.get('n_subgraphs_pred', 0) for r in successful_results]
    
    # 정확도 계산 (허용 오차 포함)
    node_accuracies = [calculate_accuracy(a, p, tolerance) for a, p in zip(actual_nodes, pred_nodes)]
    edge_accuracies = [calculate_accuracy(a, p, tolerance) for a, p in zip(actual_edges, pred_edges)]
    subgraph_accuracies = [calculate_accuracy(a, p, tolerance) for a, p in zip(actual_subgraphs, pred_subgraphs)]
    
    metrics.accuracy_nodes = sum(node_accuracies) / len(node_accuracies) * 100
    metrics.accuracy_edges = sum(edge_accuracies) / len(edge_accuracies) * 100
    metrics.accuracy_subgraphs = sum(subgraph_accuracies) / len(subgraph_accuracies) * 100
    
    # MAE 계산
    metrics.mae_nodes = calculate_mae(actual_nodes, pred_nodes)
    metrics.mae_edges = calculate_mae(actual_edges, pred_edges)
    metrics.mae_subgraphs = calculate_mae(actual_subgraphs, pred_subgraphs)
    
    # RMSE 계산
    metrics.rmse_nodes = calculate_rmse(actual_nodes, pred_nodes)
    metrics.rmse_edges = calculate_rmse(actual_edges, pred_edges)
    metrics.rmse_subgraphs = calculate_rmse(actual_subgraphs, pred_subgraphs)
    
    # 레벨 정확도 (향후 구현 가능)
    metrics.level_accuracy = 0.0  # TODO: 레벨 예측 구현 시 계산
    
    return metrics

def print_evaluation_report(metrics: EvalMetrics, results: List[Dict[str, Any]]):
    """평가 결과 출력"""
    print("\n" + "="*60)
    print("LLM Mermaid Parser 성능 평가 보고서")
    print("="*60)
    
    print(f"\n전체 통계:")
    print(f"  - 총 테스트 케이스: {len(results)}개")
    print(f"  - 성공률: {metrics.success_rate:.1f}%")
    
    successful_count = sum(1 for r in results if r['llm_ok'])
    if successful_count > 0:
        print(f"\n정확도 (허용 오차 ±1):")
        print(f"  - 노드 수 정확도: {metrics.accuracy_nodes:.1f}%")
        print(f"  - 엣지 수 정확도: {metrics.accuracy_edges:.1f}%")
        print(f"  - 서브그래프 수 정확도: {metrics.accuracy_subgraphs:.1f}%")
        
        print(f"\n평균 절대 오차 (MAE):")
        print(f"  - 노드 MAE: {metrics.mae_nodes:.2f}")
        print(f"  - 엣지 MAE: {metrics.mae_edges:.2f}")
        print(f"  - 서브그래프 MAE: {metrics.mae_subgraphs:.2f}")
        
        print(f"\n평균 제곱근 오차 (RMSE):")
        print(f"  - 노드 RMSE: {metrics.rmse_nodes:.2f}")
        print(f"  - 엣지 RMSE: {metrics.rmse_edges:.2f}")
        print(f"  - 서브그래프 RMSE: {metrics.rmse_subgraphs:.2f}")
    
    # 레벨별 분석
    level_distribution = {}
    successful_results = [r for r in results if r['llm_ok']]
    
    for result in successful_results:
        level = result.get('level', 'Unknown')
        if level not in level_distribution:
            level_distribution[level] = {'count': 0, 'node_errors': [], 'edge_errors': []}
        
        level_distribution[level]['count'] += 1
        
        # 오차 계산
        node_error = abs(result['N_actual'] - result['n_nodes_pred'])
        edge_error = abs(result['E_actual'] - result['n_edges_pred'])
        
        level_distribution[level]['node_errors'].append(node_error)
        level_distribution[level]['edge_errors'].append(edge_error)
    
    if level_distribution:
        print(f"\n레벨별 성능:")
        for level in sorted(level_distribution.keys()):
            data = level_distribution[level]
            avg_node_error = sum(data['node_errors']) / len(data['node_errors'])
            avg_edge_error = sum(data['edge_errors']) / len(data['edge_errors'])
            
            print(f"  - {level}: {data['count']}개 (노드 오차: {avg_node_error:.1f}, 엣지 오차: {avg_edge_error:.1f})")
    
    # 실패 케이스 분석
    failed_results = [r for r in results if not r['llm_ok']]
    if failed_results:
        print(f"\n실패 케이스 분석:")
        print(f"  - 실패 케이스 수: {len(failed_results)}개")
        
        failure_reasons = {}
        for result in failed_results:
            reason = "API 호출 실패" if "API 호출 실패" in result.get('llm_warnings', '') else "JSON 파싱 실패"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"    - {reason}: {count}개")

def detailed_analysis(results: List[Dict[str, Any]]):
    """상세 분석"""
    print(f"\n상세 분석:")
    
    successful_results = [r for r in results if r['llm_ok']]
    
    if not successful_results:
        print("  분석할 성공 케이스가 없습니다.")
        return
    
    # 최고/최악 성능 케이스
    node_errors = [(abs(r['N_actual'] - r['n_nodes_pred']), r['name']) for r in successful_results]
    edge_errors = [(abs(r['E_actual'] - r['n_edges_pred']), r['name']) for r in successful_results]
    
    node_errors.sort()
    edge_errors.sort()
    
    print(f"  노드 예측 최고 성능: {node_errors[0][1]} (오차: {node_errors[0][0]})")
    print(f"  노드 예측 최악 성능: {node_errors[-1][1]} (오차: {node_errors[-1][0]})")
    print(f"  엣지 예측 최고 성능: {edge_errors[0][1]} (오차: {edge_errors[0][0]})")
    print(f"  엣지 예측 최악 성능: {edge_errors[-1][1]} (오차: {edge_errors[-1][0]})")
    
    # 완벽한 예측 케이스
    perfect_cases = [r for r in successful_results 
                    if r['N_actual'] == r['n_nodes_pred'] and r['E_actual'] == r['n_edges_pred']]
    
    if perfect_cases:
        print(f"  완벽한 예측 케이스: {len(perfect_cases)}개")
        for case in perfect_cases:
            print(f"    - {case['name']}: N={case['N_actual']}, E={case['E_actual']}")

def save_evaluation_report(metrics: EvalMetrics, results: List[Dict[str, Any]], output_path: str = "evaluation_report.csv"):
    """평가 결과를 CSV로 저장"""
    try:
        # 개별 케이스 결과
        detailed_results = []
        for result in results:
            detailed_result = {
                "name": result['name'],
                "level": result.get('level', ''),
                "llm_ok": result['llm_ok'],
                "N_actual": result['N_actual'],
                "E_actual": result['E_actual'],
                "SUBGRAPH_COUNT_actual": result['SUBGRAPH_COUNT_actual'],
                "n_nodes_pred": result['n_nodes_pred'],
                "n_edges_pred": result['n_edges_pred'],
                "n_subgraphs_pred": result.get('n_subgraphs_pred', 0),
                "node_error": abs(result['N_actual'] - result['n_nodes_pred']) if result['llm_ok'] else None,
                "edge_error": abs(result['E_actual'] - result['n_edges_pred']) if result['llm_ok'] else None,
                "subgraph_error": abs(result['SUBGRAPH_COUNT_actual'] - result.get('n_subgraphs_pred', 0)) if result['llm_ok'] else None,
                "node_accurate": calculate_accuracy(result['N_actual'], result['n_nodes_pred'], 0) if result['llm_ok'] else False,
                "edge_accurate": calculate_accuracy(result['E_actual'], result['n_edges_pred'], 0) if result['llm_ok'] else False,
                "subgraph_accurate": calculate_accuracy(result['SUBGRAPH_COUNT_actual'], result.get('n_subgraphs_pred', 0), 0) if result['llm_ok'] else False,
                "warnings": result.get('llm_warnings', '')
            }
            detailed_results.append(detailed_result)
        
        # CSV 저장
        fieldnames = [
            "name", "level", "llm_ok", "N_actual", "E_actual", "SUBGRAPH_COUNT_actual",
            "n_nodes_pred", "n_edges_pred", "n_subgraphs_pred", 
            "node_error", "edge_error", "subgraph_error",
            "node_accurate", "edge_accurate", "subgraph_accurate", "warnings"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        
        print(f"\n상세 평가 결과 저장: {output_path}")
        
        # 요약 지표 저장
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary_metrics = [
            {"metric": "success_rate", "value": metrics.success_rate},
            {"metric": "accuracy_nodes", "value": metrics.accuracy_nodes},
            {"metric": "accuracy_edges", "value": metrics.accuracy_edges},
            {"metric": "accuracy_subgraphs", "value": metrics.accuracy_subgraphs},
            {"metric": "mae_nodes", "value": metrics.mae_nodes},
            {"metric": "mae_edges", "value": metrics.mae_edges},
            {"metric": "mae_subgraphs", "value": metrics.mae_subgraphs},
            {"metric": "rmse_nodes", "value": metrics.rmse_nodes},
            {"metric": "rmse_edges", "value": metrics.rmse_edges},
            {"metric": "rmse_subgraphs", "value": metrics.rmse_subgraphs}
        ]
        
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(summary_metrics)
        
        print(f"요약 지표 저장: {summary_path}")
        
    except Exception as e:
        print(f"평가 결과 저장 실패: {e}")

def main():
    """메인 함수"""
    print("LLM Mermaid Parser 평가 시작...")
    
    # LLM 결과 파일 경로
    llm_results_path = "results/SKT.csv"
    
    if not Path(llm_results_path).exists():
        print(f"LLM 결과 파일을 찾을 수 없습니다: {llm_results_path}")
        print("먼저 llm.py를 실행하여 LLM 결과를 생성하세요.")
        return
    
    # 결과 로드
    results = load_llm_results(llm_results_path)
    
    if not results:
        print("분석할 결과가 없습니다.")
        return
    
    # 성능 평가
    metrics = evaluate_performance(results, tolerance=0)
    
    # 보고서 출력
    print_evaluation_report(metrics, results)
    
    # 상세 분석
    detailed_analysis(results)
    
    # 결과 저장
    save_evaluation_report(metrics, results, "results/evaluation_report.csv")
    
    print(f"\n평가 완료!")

if __name__ == "__main__":
    main()