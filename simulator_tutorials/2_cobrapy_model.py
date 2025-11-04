"""
COBRApy 모델 사용 예제
COBRApy 라이브러리의 모델을 직접 로드하여 FBA를 수행합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # E. coli iML1515 모델 로드
    print("COBRApy에서 모델 로딩 중...")
    model = cobra.io.load_model("iML1515")
    print(f"모델명: {model.id}")
    print(f"반응 수: {len(model.reactions)}")
    print(f"대사물질 수: {len(model.metabolites)}")
    print(f"유전자 수: {len(model.genes)}")
    
    # 시뮬레이터에 모델 로드
    sim = Simulator()
    sim.load_cobra_model(model)
    
    # FBA 실행
    print("\nFBA 실행 중...")
    status, growth_rate, fluxes = sim.run_FBA()
    
    if status == 'optimal':
        print(f"\n=== FBA 결과 ===")
        print(f"최적화 상태: {status}")
        print(f"성장 속도: {growth_rate:.4f} hr⁻¹")
        
        # 주요 교환 반응 플럭스 출력
        print(f"\n주요 교환 반응:")
        exchange_reactions = ['EX_glc__D_e', 'EX_o2_e', 'EX_co2_e', 'EX_ac_e']
        for rxn_id in exchange_reactions:
            if rxn_id in fluxes:
                print(f"  {rxn_id}: {fluxes[rxn_id]:.4f}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()

