#!/usr/bin/env python3
import sys
import argparse
import cobra
from cobra import Model
from cobra.flux_analysis import gapfill as gapfill_func

def set_growth_objective_if_present(model: Model):
    """모델에 'Growth' 반응이 있으면 objective로 설정(없으면 기존 objective 유지)."""
    growth = None
    try:
        growth = model.reactions.get_by_id("Growth")
    except KeyError:
        for rxn in model.reactions:
            if rxn.id.lower() == "growth" or rxn.name.lower() == "growth":
                growth = rxn
                break
    if growth is not None:
        model.objective = growth
        print(f"[info] Objective set to: {growth.id}")
    else:
        print("[info] Using model's existing objective (no explicit 'Growth' found).")
        if model.objective:
            print(f"[info] Current objective: {model.objective}")

def check_model_feasibility(model: Model, label: str = "Model"):
    """모델의 feasibility를 체크하고 정보를 출력."""
    try:
        sol = model.optimize()
        print(f"[info] {label} is feasible. Objective value: {sol.objective_value:.6f}")
        return True
    except:
        print(f"[warning] {label} is infeasible or has no solution.")
        return False

def diagnose_model(model: Model):
    """모델의 상태를 진단."""
    print(f"\n[Diagnosis]")
    print(f"  - Reactions: {len(model.reactions)}")
    print(f"  - Metabolites: {len(model.metabolites)}")
    print(f"  - Genes: {len(model.genes)}")
    
    # Exchange reactions 확인
    exchange_rxns = [r for r in model.reactions if r.boundary]
    print(f"  - Exchange/boundary reactions: {len(exchange_rxns)}")
    
    # Objective 확인
    if model.objective:
        obj_rxns = [r.id for r in model.reactions if r.objective_coefficient != 0]
        print(f"  - Objective reactions: {obj_rxns}")
    
    # Medium 확인
    medium = model.medium
    print(f"  - Medium compounds: {len(medium)} items")
    if len(medium) < 10:  # 적은 경우만 출력
        for met, flux in list(medium.items())[:5]:
            print(f"    - {met}: {flux}")

def run_gapfilling(
    in_path: str,
    uni_path: str,
    out_path: str,
    iterations: int = 1,
    allow_demand: bool = False,
    allow_exchange: bool = False,
    produce_met: str | None = None,
    lower_bound: float = 0.05,
    penalties: dict = None,
):
    print(f"[info] Input model   : {in_path}")
    print(f"[info] Universal    : {uni_path}")
    print(f"[info] Output model : {out_path}")

    # 1) 모델 로드
    model: Model = cobra.io.read_sbml_model(in_path)
    universal: Model = cobra.io.read_sbml_model(uni_path)
    
    print(f"\n[Model Info]")
    diagnose_model(model)
    print(f"\n[Universal Model Info]")
    print(f"  - Reactions: {len(universal.reactions)}")
    print(f"  - Metabolites: {len(universal.metabolites)}")

    # 2) 초기 feasibility 체크
    if produce_met is None:
        set_growth_objective_if_present(model)
        is_feasible = check_model_feasibility(model, "Original model")
        
        if is_feasible:
            print("[warning] Model is already feasible. Gap-filling may not be necessary.")
            print("[info] Proceeding anyway to find alternative solutions...")

    # 3) Medium 설정 확인 및 조정
    if not model.medium:
        print("[warning] No medium defined. Setting minimal glucose medium...")
        # 기본 미디엄 설정 (예: 글루코스)
        try:
            model.medium = {"EX_glc__D_e": 10, "EX_o2_e": 20}
        except:
            print("[warning] Could not set default medium. Proceeding without medium constraints.")

    # 4) Gapfill 실행
    try:
        if produce_met is None:
            # Growth objective로 gapfill
            print(f"\n[info] Running gapfill with iterations={iterations}, "
                  f"demand_reactions={allow_demand}, exchange_reactions={allow_exchange}")
            print(f"[info] Lower bound for objective: {lower_bound}")
            
            solutions = gapfill_func(
                model,
                universal,
                demand_reactions=allow_demand,
                exchange_reactions=allow_exchange,
                iterations=iterations,
                lower_bound=lower_bound,  # objective의 최소값
                penalties=penalties,  # 반응별 penalty (없으면 기본값 사용)
            )
        else:
            # 특정 대사체 생산을 위한 gapfill
            print(f"\n[info] Using demand objective for metabolite: {produce_met}")
            if produce_met not in model.metabolites:
                # 대사체 ID 목록 일부 출력
                met_ids = [m.id for m in model.metabolites]
                print(f"[error] Metabolite '{produce_met}' not found.")
                print(f"[info] Example metabolite IDs: {met_ids[:10]}")
                raise KeyError(f"Metabolite '{produce_met}' not found in model.")
            
            with model:
                dm_rxn = model.add_boundary(
                    model.metabolites.get_by_id(produce_met), 
                    type="demand"
                )
                model.objective = dm_rxn
                print(f"[info] Temporary objective: {dm_rxn.id}")
                
                solutions = gapfill_func(
                    model,
                    universal,
                    demand_reactions=allow_demand,
                    exchange_reactions=allow_exchange,
                    iterations=iterations,
                    lower_bound=lower_bound,
                    penalties=penalties,
                )

    except cobra.exceptions.Infeasible as e:
        print(f"\n[error] Gap-filling failed: {e}")
        print("\n[Troubleshooting suggestions:]")
        print("  1. Check if the universal model contains necessary reactions")
        print("  2. Try with --allow-exchange and/or --allow-demand flags")
        print("  3. Reduce --lower-bound value (e.g., 0.01 or 0.001)")
        print("  4. Verify that the medium is properly defined")
        print("  5. Check if objective reaction exists and is correctly defined")
        sys.exit(2)
    except Exception as e:
        print(f"\n[error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

    if not solutions:
        print("[warning] No gap-filling solution found.")
        print("[info] This might mean the model cannot be made feasible with available reactions.")
        sys.exit(2)

    # 5) 솔루션 적용
    print(f"\n[info] Found {len(solutions)} solution(s).")
    
    for i, solution in enumerate(solutions, 1):
        rxn_list = list(solution)
        print(f"\n[Solution #{i}] {len(rxn_list)} reactions to add:")
        for rxn in rxn_list[:10]:  # 처음 10개만 출력
            print(f"  + {rxn.id}: {rxn.name if rxn.name else 'no name'}")
        if len(rxn_list) > 10:
            print(f"  ... and {len(rxn_list) - 10} more reactions")

    # 첫 번째 솔루션 적용
    first_solution = list(solutions[0])
    model.add_reactions(first_solution)

    # 6) 결과 확인
    print("\n[info] Verifying gap-filled model...")
    sol = model.optimize()
    print(f"[info] Objective value after gap-filling: {sol.objective_value:.6f}")
    
    # 추가된 반응 중 실제로 flux를 가지는 반응 확인
    active_added = [r.id for r in first_solution if abs(sol.fluxes[r.id]) > 1e-6]
    print(f"[info] Active gap-filled reactions: {len(active_added)}/{len(first_solution)}")
    for rxn_id in active_added[:5]:
        print(f"  - {rxn_id}: flux = {sol.fluxes[rxn_id]:.4f}")

    # 7) 저장
    cobra.io.write_sbml_model(model, out_path)
    print(f"\n[info] Saved gap-filled model -> {out_path}")

def build_parser():
    p = argparse.ArgumentParser(
        description="COBRApy gap-filling script with enhanced error handling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_model", help="입력 SBML 모델 경로")
    p.add_argument("universal_model", help="유니버설 SBML 모델 경로")
    p.add_argument("output_model", help="gap-fill 후 저장할 SBML 경로")
    p.add_argument("--iterations", type=int, default=1,
                   help="솔루션 탐색 반복 횟수")
    p.add_argument("--allow-demand", action="store_true",
                   help="demand 반응 허용")
    p.add_argument("--allow-exchange", action="store_true",
                   help="exchange 반응 허용")
    p.add_argument("--produce-met", type=str, default=None,
                   help="특정 대사체 생산 목표 (예: atp_c)")
    p.add_argument("--lower-bound", type=float, default=0.05,
                   help="objective의 최소 요구값")
    return p

def main():
    args = build_parser().parse_args()
    
    # penalties 예시 (필요시 사용)
    # penalties = {"EX_o2_e": 0, "EX_glc__D_e": 0}  # 이 반응들은 penalty 없음
    
    run_gapfilling(
        in_path=args.input_model,
        uni_path=args.universal_model,
        out_path=args.output_model,
        iterations=args.iterations,
        allow_demand=args.allow_demand,
        allow_exchange=args.allow_exchange,
        produce_met=args.produce_met,
        lower_bound=args.lower_bound,
        penalties=None,  # 또는 특정 penalties dict 전달
    )

if __name__ == "__main__":
    main()
