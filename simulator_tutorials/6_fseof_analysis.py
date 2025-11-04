"""
FSEOF를 사용한 생산 Envelope 분석
제약된 성장 조건에서 목표 대사물질의 생산 범위를 탐색합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 먼저 최대 성장 속도 확인
    print("\n=== 야생형 FBA 실행 ===")
    _, max_growth, _ = sim.run_FBA()
    print(f"최대 성장 속도: {max_growth:.4f} hr⁻¹")
    
    # 숙신산 생산을 위한 FSEOF 분석
    target_reaction = 'EX_succ_e'  # 숙신산 분비 반응
    objective_fraction = 0.9  # 최대 성장의 90%
    
    print(f"\n=== FSEOF 분석: {target_reaction} ===")
    print(f"성장 제약: 최대 성장의 {objective_fraction*100:.0f}%")
    
    # FSEOF 실행
    status, min_flux, max_flux, flux_range = sim.run_FSEOF(
        target_reaction=target_reaction,
        objective_fraction=objective_fraction
    )
    
    if status == 'optimal':
        print(f"\n최적화 상태: {status}")
        print(f"제약된 성장 속도: {max_growth * objective_fraction:.4f} hr⁻¹")
        print(f"\n숙신산 생산 범위:")
        print(f"  최소 생산: {min_flux:.4f} mmol/gDW/hr")
        print(f"  최대 생산: {max_flux:.4f} mmol/gDW/hr")
        print(f"  생산 범위: {max_flux - min_flux:.4f} mmol/gDW/hr")
        
        # 최대 생산 시 성장률 확인
        print(f"\n최대 생산 달성 가능 여부:")
        if max_flux > 1e-6:
            print(f"  성장을 {objective_fraction*100:.0f}% 유지하면서 최대 {max_flux:.4f} mmol/gDW/hr 생산 가능")
        else:
            print(f"  해당 성장 조건에서는 숙신산 생산 불가")
    else:
        print(f"최적화 실패: {status}")
        print("해당 조건에서는 생산이 불가능합니다.")

if __name__ == "__main__":
    main()

