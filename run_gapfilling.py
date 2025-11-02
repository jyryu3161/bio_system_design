#!/usr/bin/env python3
import sys
import cobra
from cobra import Model, Reaction
from cobra.flux_analysis.gapfilling import GapFiller

def set_growth_objective(model: Model):
    """Objective를 Growth로 설정(가능하면). 없으면 기존 objective 유지."""
    growth_rxn = None
    try:
        growth_rxn = model.reactions.get_by_id("Growth")
    except KeyError:
        for rxn in model.reactions:
            if rxn.id.lower() == "growth" or rxn.name.lower() == "growth":
                growth_rxn = rxn
                break
    if growth_rxn is not None:
        model.objective = growth_rxn
        print(f"[info] Objective set to: {growth_rxn.id}")
    else:
        print("[warn] 'Growth' reaction not found. Keep existing objective.")

def main():
    if len(sys.argv) != 4:
        print("사용법: python gapfill.py <input_model.xml> <universal_model.xml> <output_model.xml>")
        sys.exit(1)

    in_path, uni_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"[info] Input model   : {in_path}")
    print(f"[info] Universal    : {uni_path}")
    print(f"[info] Output model : {out_path}")

    # 1) 모델 로드
    model: Model = cobra.io.read_sbml_model(in_path)
    universal: Model = cobra.io.read_sbml_model(uni_path)

    # 2) Objective 설정 (선택)
    set_growth_objective(model)

    # (선택) 필요 시 여기서 미디어를 설정하세요:
    # model.medium = {...}

    # 3) GapFiller 설정
    # - demand_reactions / exchange_reactions: True로 하면 필요 시 수요/교환반응도 후보에 포함
    # - penalties: 반응별 비용(낮을수록 추가가 쉬움). 간단히 None으로 두면 동일 비용.
    gapfiller = GapFiller(
        model=model,
        universal=universal,
        demand_reactions=True,
        exchange_reactions=True,
        # penalties=your_penalty_dict   # 필요 시 주석 해제하여 가중치 설정
    )

    # 4) Gap-fill 실행 (한 개 해 찾기)
    print("[info] Running gap-filling (searching for 1 solution)...")
    solutions = gapfiller.fill(n_solutions=1)
    if not solutions:
        print("[error] No gap-filling solution found. Check medium/objective/universal model.")
        sys.exit(2)

    # solutions는 보통 '추가해야 할 반응들의 리스트(set)'를 요소로 가진 리스트입니다.
    # 여기서는 첫 번째 해를 사용
    solution_reactions = list(solutions[0])
    print(f"[info] Reactions to add: {len(solution_reactions)}")

    # 5) 반응 추가
    # universal의 Reaction 객체를 그대로 모델에 추가합니다.
    # (필요 시 deepcopy하여 복제할 수도 있으나 COBRApy가 모델 재할당을 처리합니다.)
    model.add_reactions(solution_reactions)

    # 6) 최적화로 성장 확인
    sol = model.optimize()
    print("[info] Optimization done.")
    print(f"      objective value (growth): {sol.objective_value:.6f}")

    # 7) 저장
    cobra.io.write_sbml_model(model, out_path)
    print(f"[info] Saved gap-filled model -> {out_path}")

if __name__ == "__main__":
    main()
