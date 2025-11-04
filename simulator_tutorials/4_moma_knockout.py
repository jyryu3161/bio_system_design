"""
MOMA를 사용한 유전자 Knockout 시뮬레이션
유전자 knockout 후 대사 플럭스 변화를 최소화하는 방식으로 표현형을 예측합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 야생형 (참조) 플럭스 분포 획득
    print("\n=== 야생형 FBA 실행 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    
    # PGI 유전자 knockout 시뮬레이션
    print("\n=== PGI 유전자 Knockout (MOMA) ===")
    knockout_constraints = {'PGI': (0, 0)}  # PGI 반응 차단
    
    # MOMA 실행하여 knockout 표현형 예측
    status, distance, moma_fluxes = sim.run_MOMA(
        wild_flux=wt_fluxes,
        flux_constraints=knockout_constraints
    )
    
    if status == 'optimal':
        knockout_growth = moma_fluxes[sim.objective]
        print(f"최적화 상태: {status}")
        print(f"MOMA 예측 성장 속도: {knockout_growth:.4f} hr⁻¹")
        print(f"성장 감소: {(1 - knockout_growth/wt_growth)*100:.1f}%")
        print(f"대사 거리 (L1 norm): {distance:.4f}")
        
        # 플럭스 변화가 큰 반응 찾기
        flux_changes = {
            rxn: abs(moma_fluxes[rxn] - wt_fluxes[rxn]) 
            for rxn in wt_fluxes.keys()
        }
        top_changes = sorted(flux_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n가장 많이 변화된 상위 5개 반응:")
        for rxn, change in top_changes:
            print(f"  {rxn}: {wt_fluxes[rxn]:.4f} → {moma_fluxes[rxn]:.4f} (변화: {change:.4f})")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()

