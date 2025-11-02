import cobra
import sys
from cobra import Model

# === 0) 실행 인자 확인 ===
if len(sys.argv) < 2:
    print("사용법: python run.py <model_file>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"입력 모델 파일: {model_path}")

# === 1) 교환반응 자동 찾기 함수 ===
def find_exchange_for_met(model: Model, met_core: str):
    """
    met_core 예: 'glc__D', 'nh4', 'ca2' ...
    교환반응은 보통 EX_<met>_e 형태이거나, 반응 내 외부(e) metabolite를 포함.
    """
    candidate_ids = [
        f"EX_{met_core}_e",
        f"EX_{met_core}(e)",
        f"EX_{met_core}__e",
        f"EX_{met_core}e",
    ]
    for cid in candidate_ids:
        try:
            return model.reactions.get_by_id(cid)
        except KeyError:
            pass

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


# === 2) 모델 불러오기 ===
model: Model = cobra.io.read_sbml_model(model_path)

# === 3) Objective 설정 ===
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

# === 4) M9 배지 조건 ===
m9_targets = {
    "glc__D": -10.0,  # D-Glucose
    "nh4":    -10.0,  # Ammonium
    "pi":     -10.0,  # Phosphate
    "so4":    -10.0,  # Sulfate
    "o2":     -20.0,  # Oxygen
    "h2o":  -1000.0,  # Water
    "h":    -1000.0,  # Proton
    "na1":  -1000.0,  # Na+
    "k":    -1000.0,  # K+
    "cl":   -1000.0,  # Cl-
    "ca2":  -1000.0,
    "cobalt2": -1000.0,
    "cu2":  -1000.0,
    "fe2":  -1000.0,
    "fe3":  -1000.0,
    "mg2":  -1000.0,
    "mn2":  -1000.0,
    "mobd": -1000.0,
    "ni2":  -1000.0,
    "zn2":  -1000.0,
}

# === 5) 모델 배지 구성 ===
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

# === 6) 출력 ===
print("===== M9 Medium Mapping =====")
for met_core, rxn_id, uptake in mapped:
    print(f"{met_core:10s} -> {rxn_id:25s} : {uptake}")

if not_found:
    print("\n[주의] 다음 성분의 교환반응을 찾지 못했습니다:")
    print(", ".join(not_found))

# === 7) 최적화 및 결과 출력 ===
solution = model.optimize()
print("\n===== Growth Result under M9 =====")
print(f"Objective (growth rate): {solution.objective_value:.6f}")

# === 8) 모델 저장 ===
output_path = "new_model_M9_biomass.xml"
cobra.io.write_sbml_model(model, output_path)
print(f"\nSaved: {output_path}")
