import sys
import cobra
from cobra import Model

# === 0) 실행 인자 처리 ===
if len(sys.argv) < 3:
    print("사용법: python run.py <input_model.xml> <output_model.xml>")
    sys.exit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]
print(f"입력 모델: {in_path}")
print(f"출력 모델: {out_path}")

# === 편의 함수: 교환반응 자동 찾기 ===
def find_exchange_for_met(model: Model, met_core: str):
    """
    met_core 예: 'glc__D', 'nh4', 'ca2' ...
    교환반응은 보통 EX_<met>_e 형태이거나, 반응 내 외부(e) metabolite를 포함.
    """
    candidate_ids = [
        f"EX_{met_core}_e",
        f"EX_{met_core}(e)",   # 일부 모델 관습
        f"EX_{met_core}__e",   # 드물지만 대비
        f"EX_{met_core}e",     # 드물지만 대비
    ]
    for cid in candidate_ids:
        try:
            return model.reactions.get_by_id(cid)
        except KeyError:
            pass

    # 반응 내 metabolite를 보고 탐색
    for rxn in model.exchanges:
        for m in rxn.metabolites:
            mid_norm = (
                m.id.replace('[e]', '')
                 .replace('(e)', '')
                 .replace('_e', '')
            )
            if mid_norm == met_core:
                return rxn
    return None

# === 1) 모델 불러오기 ===
model: Model = cobra.io.read_sbml_model(in_path)

# === 2) Objective를 Growth로 설정(가능하면) ===
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
else:
    print("[경고] 'Growth' 반응을 찾지 못했습니다. Objective는 기존 값으로 둡니다.")

# === 3) M9 성분 리스트와 uptake 설정 ===
m9_targets = {
    "glc__D": -10.0,  # D-Glucose (탄소원)
    "nh4":    -10.0,  # Ammonium (질소)
    "pi":     -10.0,  # Phosphate (인)
    "so4":    -10.0,  # Sulfate (황)
    "o2":     -20.0,  # Oxygen
    "h2o":  -1000.0,  # Water
    "h":    -1000.0,  # Proton
    "na1":  -1000.0,  # Na+
    "k":    -1000.0,  # K+
    "cl":   -1000.0,  # Cl-
    # 금속/미량원소들
    "ca2":  -1000.0,
    "cobalt2": -1000.0,
    "cu2":  -1000.0,
    "fe2":  -1000.0,
    "fe3":  -1000.0,
    "mg2":  -1000.0,
    "mn2":  -1000.0,
    "mobd": -1000.0,  # molybdate
    "ni2":  -1000.0,
    "zn2":  -1000.0,
}

# === 4) 모델 배지 구성 ===
new_medium = model.medium.copy()
not_found = []
mapped = []

for met_core, uptake in m9_targets.items():
    rxn = find_exchange_for_met(model, met_core)
    if rxn is None:
        not_found.append(met_core)
        continue
    new_medium[rxn.id] = uptake
    mapped.append((met_core, rxn.id, uptake))

model.medium = new_medium

# === 5) 결과 출력 ===
print("===== M9 Medium Mapping =====")
for met_core, rxn_id, uptake in mapped:
    print(f"{met_core:10s} -> {rxn_id:25s} : {uptake}")

if not_found:
    print("\n[주의] 다음 성분의 교환반응을 찾지 못했습니다. 모델의 ID를 확인하세요:")
    print(", ".join(not_found))

# === 6) 최적화 및 성장률 확인 ===
solution = model.optimize()
print("\n===== Growth Result under M9 =====")
print(f"Objective (growth rate): {solution.objective_value:.6f}")

# === 7) 저장 (출력 경로 사용) ===
cobra.io.write_sbml_model(model, out_path)
print(f"\nSaved: {out_path}")
