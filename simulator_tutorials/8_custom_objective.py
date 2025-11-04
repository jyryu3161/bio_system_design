"""
사용자 정의 목적 함수 예제
특정 대사물질의 생산을 최대화하거나 다른 목적 함수를 사용합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 기본 성장 최대화 (바이오매스)
    print("\n=== 기본 목적 함수: 성장 최대화 ===")
    status, growth_rate, fluxes = sim.run_FBA()
    print(f"최대 성장 속도: {growth_rate:.4f} hr⁻¹")
    print(f"숙신산 생산: {fluxes.get('EX_succ_e', 0):.4f} mmol/gDW/hr")
    
    # 숙신산 생산 최대화
    print("\n=== 사용자 정의 목적 함수: 숙신산 생산 최대화 ===")
    status, production_rate, fluxes = sim.run_FBA(
        new_objective='EX_succ_e',
        mode='max'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]  # 원래 목적 함수 (바이오매스) 값
        print(f"최대 숙신산 생산: {production_rate:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")
    
    # 아세트산 생산 최대화
    print("\n=== 사용자 정의 목적 함수: 아세트산 생산 최대화 ===")
    status, production_rate, fluxes = sim.run_FBA(
        new_objective='EX_ac_e',
        mode='max'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]
        print(f"최대 아세트산 생산: {production_rate:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")
    
    # ATP 유지 비용 최소화
    print("\n=== 사용자 정의 목적 함수: ATP 유지 비용 최소화 ===")
    status, atp_cost, fluxes = sim.run_FBA(
        new_objective='ATPM',
        mode='min'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]
        print(f"최소 ATP 유지 비용: {atp_cost:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")

if __name__ == "__main__":
    main()

