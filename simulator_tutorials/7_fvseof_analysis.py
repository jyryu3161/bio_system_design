"""
FVSEOF를 사용한 대사 유연성 분석
여러 생산 수준에서 대사 네트워크의 유연성을 종합적으로 분석합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 최대 성장 속도 확인
    print("\n=== 야생형 FBA 실행 ===")
    _, max_growth, _ = sim.run_FBA()
    print(f"최대 성장 속도: {max_growth:.4f} hr⁻¹")
    
    # 숙신산 생산에 대한 FVSEOF 분석
    target_reaction = 'EX_succ_e'
    objective_fraction = 0.9
    num_points = 10
    
    print(f"\n=== FVSEOF 분석: {target_reaction} ===")
    print(f"성장 제약: 최대 성장의 {objective_fraction*100:.0f}%")
    print(f"분석 포인트: {num_points}개")
    
    # FVSEOF 실행
    status, fvseof_results = sim.run_FVSEOF(
        target_reaction=target_reaction,
        objective_fraction=objective_fraction,
        num_points=num_points
    )
    
    if status == 'optimal':
        print(f"\n최적화 상태: {status}")
        print(f"분석된 포인트 수: {len(fvseof_results['target_fluxes'])}")
        
        print(f"\n{'포인트':<8} {'생산량':<12} {'최소 성장':<12} {'최대 성장':<12} {'성장 범위':<12}")
        print("=" * 60)
        
        # 각 포인트에서 대사 유연성 분석
        for i, target_flux in enumerate(fvseof_results['target_fluxes']):
            obj_flux = fvseof_results['objective_fluxes'][i]
            min_growth = obj_flux['minimum']
            max_growth = obj_flux['maximum']
            growth_range = max_growth - min_growth
            
            print(f"{i+1:<8} {target_flux:<12.4f} {min_growth:<12.4f} {max_growth:<12.4f} {growth_range:<12.4f}")
        
        # 요약 통계
        print(f"\n=== 요약 ===")
        print(f"최소 생산량: {min(fvseof_results['target_fluxes']):.4f}")
        print(f"최대 생산량: {max(fvseof_results['target_fluxes']):.4f}")
        
        growth_ranges = [
            fvseof_results['objective_fluxes'][i]['maximum'] - 
            fvseof_results['objective_fluxes'][i]['minimum']
            for i in range(len(fvseof_results['target_fluxes']))
        ]
        avg_flexibility = sum(growth_ranges) / len(growth_ranges)
        print(f"평균 성장 유연성: {avg_flexibility:.4f} hr⁻¹")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()

