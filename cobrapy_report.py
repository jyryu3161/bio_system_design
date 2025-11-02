import cobra
from cobra import Model
from cobra.util.solver import linear_reaction_coefficients

# --- 1. 모델 불러오기 ---
model: Model = cobra.io.read_sbml_model("./new_model.xml")

# --- 2. Growth 반응을 objective로 설정 ---
# 우선적으로 ID가 'Growth'인 반응을 찾고, 없으면 이름/부분일치로 탐색
growth_rxn = None
try:
    growth_rxn = model.reactions.get_by_id("Growth")
except KeyError:
    # 이름이 정확히 'Growth'인 경우
    for rxn in model.reactions:
        if rxn.name == "Growth":
            growth_rxn = rxn
            break
    # 소문자 부분일치(백업)
    if growth_rxn is None:
        for rxn in model.reactions:
            if ("growth" in rxn.id.lower()) or ("growth" in rxn.name.lower()):
                growth_rxn = rxn
                break

if growth_rxn is None:
    raise ValueError("Growth(성장) 반응을 찾지 못했습니다. 모델 내 Growth/BIOMASS 계열 반응 ID를 확인하세요.")

# objective을 Growth로 설정 (기본 방향은 'max')
model.objective = growth_rxn

# --- 3. 모델 기본 정보 출력 ---
print("===== Model Basic Information =====")
print(f"Number of reactions   : {len(model.reactions)}")
print(f"Number of metabolites : {len(model.metabolites)}")
print(f"Number of genes       : {len(model.genes)}")
print()

# --- 4. Objective 정보 출력 (안전한 방식) ---
print("===== Objective Information =====")
print(f"Objective expression   : {model.objective.expression}")
print(f"Objective direction    : {model.objective.direction}")

# 각 반응별 objective 계수 표시 (0이 아닌 것만)
print("Objective coefficients (by reaction):")
for rxn in model.reactions:
    if rxn.objective_coefficient != 0:
        print(f"  {rxn.id}: {rxn.objective_coefficient}")
print()

# --- 5. 배지(Exchange Reactions) 정보 ---
print("===== Medium (Exchange Reactions) =====")
medium = model.medium
for rxn_id, rate in medium.items():
    print(f"{rxn_id:25s} uptake rate: {rate}")
print()

# --- 6. 최적화 및 growth rate 출력 ---
solution = model.optimize()
print("===== Growth Rate =====")
print(f"Objective value (growth rate): {solution.objective_value:.4f}")
print()

# --- 7. (선택) 비영(0이 아닌) flux 확인 ---
print("===== Top Fluxes (non-zero) =====")
for rxn_id, flux in solution.fluxes.items():
    if abs(flux) > 1e-6:
        print(f"{rxn_id:25s} : {flux:.4f}")

# --- 8. Objective가 반영된 모델을 SBML로 저장 ---
cobra.io.write_sbml_model(model, "new_model_biomass.xml")
print("\nSaved: new_model_biomass.xml")
