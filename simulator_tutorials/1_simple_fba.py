"""
기본 FBA 분석 예제
Flux Balance Analysis를 수행하여 최적 성장 속도를 예측합니다.
"""

from Simulator import Simulator

def main():
    # 시뮬레이터 초기화
    sim = Simulator()
    
    # 대사 모델 로드 (SBML 형식)
    print("모델 로딩 중...")
    sim.read_model('iML1515.xml')
    
    # FBA 실행
    print("\nFBA 실행 중...")
    status, objective_value, flux_distribution = sim.run_FBA()
    
    if status == 'optimal':
        print(f"\n=== FBA 결과 ===")
        print(f"최적화 상태: {status}")
        print(f"최적 성장 속도: {objective_value:.4f} hr⁻¹")
        print(f"활성 반응 수: {sum(1 for v in flux_distribution.values() if abs(v) > 1e-6)}")
        print(f"전체 반응 수: {len(flux_distribution)}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()

