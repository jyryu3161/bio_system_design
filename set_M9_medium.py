import cobra

# --- 1. 모델 불러오기 ---
model = cobra.io.read_sbml_model("new_model.xml")

# --- 2. 기본 M9 배지 구성 ---
# M9 minimal medium 구성 성분 (E. coli 기준)
M9_medium = {
    'EX_glc__D_e': -10.0,   # glucose uptake
    'EX_nh4_e': -10.0,      # ammonium
    'EX_pi_e': -10.0,       # phosphate
    'EX_so4_e': -10.0,      # sulfate
    'EX_o2_e': -20.0,       # oxygen
    'EX_h2o_e': -1000.0,    # water
    'EX_na1_e': -1000.0,    # sodium
    'EX_cl_e': -1000.0,     # chloride
    'EX_k_e': -1000.0,      # potassium
    'EX_h_e': -1000.0,      # proton
    'EX_co2_e': -1000.0,    # carbon dioxide
}

# --- 3. 모델의 배지를 M9으로 설정 ---
model.medium = M9_medium

# --- 4. 설정 확인 ---
print("===== Medium Set to M9 =====")
for rxn_id, rate in model.medium.items():
    print(f"{rxn_id:20s} : {rate}")

# --- 5. Growth 최적화 ---
solution = model.optimize()
print("\n===== Growth Result =====")
print(f"Objective value (growth rate): {solution.objective_value:.4f}")

# --- 6. M9 설정된 모델 저장 ---
cobra.io.write_sbml_model(model, "new_model_M9.xml")
print("\nSaved model with M9 medium: new_model_M9.xml")
