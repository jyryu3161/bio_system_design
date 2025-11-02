import cobra

# --- 1. 모델 불러오기 ---
# 예: XML(SBML) 형식의 모델 불러오기
# (경로는 사용자 모델 파일 경로로 변경)
model = cobra.io.read_sbml_model("path/to/model.xml")

# --- 2. 모델 기본 정보 출력 ---
print("===== Model Basic Information =====")
print(f"Number of reactions   : {len(model.reactions)}")
print(f"Number of metabolites : {len(model.metabolites)}")
print(f"Number of genes       : {len(model.genes)}")
print()

# --- 3. Objective Reaction ID 확인 ---
print("===== Objective Information =====")
print(f"Objective Reaction ID : {model.objective.expression}")
print(f"Objective Coefficients: {model.objective_coefficients}")
print()

# --- 4. 배지(Exchange Reactions) 정보 확인 ---
# Exchange reaction은 일반적으로 "EX_" 로 시작하거나 bounds에 open boundary가 있는 반응임
print("===== Medium (Exchange Reactions) =====")
medium = model.medium  # 현재 설정된 배지 (dict 형태: {reaction_id: uptake_rate})
for rxn_id, rate in medium.items():
    print(f"{rxn_id:25s} uptake rate: {rate}")
print()

# --- 5. Growth Rate (Objective 값) 계산 ---
solution = model.optimize()
print("===== Growth Rate =====")
print(f"Objective value (growth rate): {solution.objective_value:.4f}")
print()

# --- 6. (선택) 성장에 기여하는 주요 flux 확인 ---
print("===== Top Fluxes (non-zero) =====")
for rxn, flux in solution.fluxes.items():
    if abs(flux) > 1e-6:
        print(f"{rxn:25s} : {flux:.4f}")
