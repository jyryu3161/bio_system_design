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

def run_gapfilling(
    in_path: str,
    uni_path: str,
    out_path: str,
    iterations: int = 1,
    allow_demand: bool = False,
    allow_exchange: bool = False,
    produce_met: str | None = None,
):
    print(f"[info] Input model   : {in_path}")
    print(f"[info] Universal    : {uni_path}")
    print(f"[info] Output model : {out_path}")

    # 1) 모델 로드
    model: Model = cobra.io.read_sbml_model(in_path)
    universal: Model = cobra.io.read_sbml_model(uni_path)

    # 2) Objective: 기본은 모델의 objective(있으면 Growth로 세팅)
    #    단, --produce-met가 있으면 demand boundary를 objective로 사용
    if produce_met is None:
        set_growth_objective_if_present(model)

        # 3) gapfill 실행 (manual 예시: solution = gapfill(model, universal, demand_reactions=False))
        print(f"[info] Running gapfill with iterations={iterations}, "
              f"demand_reactions={allow_demand}, exchange_reactions={allow_exchange}")
        solutions = gapfill_func(
            model,
            universal,
            demand_reactions=allow_demand,
            exchange_reactions=allow_exchange,
            iterations=iterations,
        )
    else:
        # 매뉴얼 예시처럼 context 내에서 demand boundary를 objective로 사용
        print(f"[info] Using demand objective for metabolite: {produce_met}")
        if produce_met not in model.metabolites:
            raise KeyError(
                f"Metabolite '{produce_met}' not found in model. "
                f"Check metabolite ID (e.g., 'f6p_c')."
            )
        with model:
            dm_rxn = model.add_boundary(model.metabolites.get_by_id(produce_met), type="demand")
            model.objective = dm_rxn
            print(f"[info] Temporary objective set to DEMAND({produce_met}) -> {dm_rxn.id}")
            print(f"[info] Running gapfill with iterations={iterations}, "
                  f"demand_reactions={allow_demand}, exchange_reactions={allow_exchange}")
            solutions = gapfill_func(
                model,
                universal,
                demand_reactions=allow_demand,
                exchange_reactions=allow_exchange,
                iterations=iterations,
            )

    if not solutions:
        print("[error] No gap-filling solution(s) returned. "
              "Check medium/objective/universal model.")
        sys.exit(2)

    # 여러 해가 있을 수 있으나, 일단 첫 번째 해를 적용
    first_solution = list(solutions[0])
    print(f"[info] Solution #1 reactions to add: {len(first_solution)}")
    for rxn in first_solution:
        print(f"  + {rxn.id}")

    # 4) 실제 모델에 반영
    # universal에서 가져온 반응 객체를 그대로 추가
    model.add_reactions(first_solution)

    # 5) 최적화로 확인
    sol = model.optimize()
    print(f"[info] Optimization after adding solution #1 -> objective: {sol.objective_value:.6f}")

    # 6) 저장
    cobra.io.write_sbml_model(model, out_path)
    print(f"[info] Saved gap-filled model -> {out_path}")

def build_parser():
    p = argparse.ArgumentParser(
        description="COBRApy gap-filling (manual-style) script:\n"
                    " - 기본은 모델의 objective(가능하면 Growth)를 사용\n"
                    " - --produce-met 지정 시 해당 대사체 생산을 위한 demand를 objective로 사용",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_model", help="입력 SBML 모델 경로")
    p.add_argument("universal_model", help="유니버설(추가 후보 반응) SBML 모델 경로")
    p.add_argument("output_model", help="gap-fill 적용 후 저장할 SBML 경로")
    p.add_argument("--iterations", type=int, default=1,
                   help="여러 솔루션 탐색 횟수 (manual: iterations=4 예시)")
    p.add_argument("--allow-demand", action="store_true",
                   help="수요(demand) 반응도 후보에 포함")
    p.add_argument("--allow-exchange", action="store_true",
                   help="교환(exchange) 반응도 후보에 포함")
    p.add_argument("--produce-met", type=str, default=None,
                   help="이 대사체를 생산 가능하도록 demand boundary를 objective로 사용 (예: f6p_c)")
    return p

def main():
    args = build_parser().parse_args()
    run_gapfilling(
        in_path=args.input_model,
        uni_path=args.universal_model,
        out_path=args.output_model,
        iterations=args.iterations,
        allow_demand=args.allow-demand if hasattr(args, "allow-demand") else args.allow_demand,
        allow_exchange=args.allow_exchange,
        produce_met=args.produce_met,
    )

if __name__ == "__main__":
    main()
