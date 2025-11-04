"""
ROOM을 사용한 유전자 Knockout 시뮬레이션
유의미하게 변화된 반응의 수를 최소화하여 knockout 표현형을 예측합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 야생형 플럭스 분포 획득
    print("\n=== 야생형 FBA 실행 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    
    # PGI 유전자 knockout 시뮬레이션
    print("\n=== PGI 유전자 Knockout (ROOM) ===")
    knockout_constraints = {'PGI': (0, 0)}
    
    # ROOM 실행하여 knockout 표현형 예측
    status, n_changed, room_fluxes = sim.run_ROOM(
        wild_flux=wt_fluxes,
        flux_constraints=knockout_constraints,
        delta=0.03,  # 3% 상대 허용오차
        epsilon=0.001  # 0에 가까운 플럭스의 절대 허용오차
    )
    
    if status == 'optimal':
        knockout_growth = room_fluxes[sim.objective]
        print(f"최적화 상태: {status}")
        print(f"ROOM 예측 성장 속도: {knockout_growth:.4f} hr⁻¹")
        print(f"성장 감소: {(1 - knockout_growth/wt_growth)*100:.1f}%")
        print(f"유의미하게 변화된 반응 수: {int(n_changed)}")
        
        # 변화된 반응 찾기 (delta와 epsilon 기준 적용)
        changed_reactions = []
        for rxn in wt_fluxes.keys():
            wt_val = abs(wt_fluxes[rxn])
            room_val = abs(room_fluxes[rxn])
            
            # ROOM의 변화 기준 적용
            threshold = max(0.03 * wt_val, 0.001)
            if abs(room_val - wt_val) > threshold:
                changed_reactions.append((rxn, wt_fluxes[rxn], room_fluxes[rxn]))
        
        print(f"\n유의미하게 변화된 반응 (상위 5개):")
        for rxn, wt_val, ko_val in changed_reactions[:5]:
            print(f"  {rxn}: {wt_val:.4f} → {ko_val:.4f}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()

