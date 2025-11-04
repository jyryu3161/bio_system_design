"""
Parsimonious FBA (pFBA) 예제
최적 성장을 유지하면서 총 플럭스를 최소화합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 표준 FBA 실행
    print("\n=== 표준 FBA 실행 ===")
    status_fba, growth_fba, fluxes_fba = sim.run_FBA()
    
    if status_fba == 'optimal':
        total_flux_fba = sum(abs(v) for v in fluxes_fba.values())
        active_rxns_fba = sum(1 for v in fluxes_fba.values() if abs(v) > 1e-6)
        
        print(f"성장 속도: {growth_fba:.4f} hr⁻¹")
        print(f"총 플럭스 합계: {total_flux_fba:.4f}")
        print(f"활성 반응 수: {active_rxns_fba}")
    
    # Parsimonious FBA 실행
    print("\n=== Parsimonious FBA (pFBA) 실행 ===")
    status_pfba, total_flux_pfba, fluxes_pfba = sim.run_FBA(
        internal_flux_minimization=True
    )
    
    if status_pfba == 'optimal':
        growth_pfba = fluxes_pfba[sim.objective]
        active_rxns_pfba = sum(1 for v in fluxes_pfba.values() if abs(v) > 1e-6)
        
        print(f"성장 속도: {growth_pfba:.4f} hr⁻¹")
        print(f"총 플럭스 합계: {total_flux_pfba:.4f}")
        print(f"활성 반응 수: {active_rxns_pfba}")
        
        # 비교
        print(f"\n=== FBA vs pFBA 비교 ===")
        print(f"총 플럭스 감소: {total_flux_fba - total_flux_pfba:.4f} ({(1 - total_flux_pfba/total_flux_fba)*100:.1f}%)")
        print(f"활성 반응 감소: {active_rxns_fba - active_rxns_pfba} 개")
    else:
        print(f"최적화 실패: {status_pfba}")

if __name__ == "__main__":
    main()

