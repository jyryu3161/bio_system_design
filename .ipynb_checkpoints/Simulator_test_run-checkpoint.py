from Simulator import Simulator
from cobra.io import load_model

model = load_model("textbook")

# 모델 로드
sim = Simulator()
sim.load_cobra_model(model)

# 1. FBA 실행
status, obj_val, flux_dist = sim.run_FBA()

# 2. 유전자 녹아웃 시뮬레이션 (wild_flux를 FBA 결과로)
knockout_constraints = {'PGI': (0, 0)}  # PGI 반응 녹아웃

# MOMA로 예측
status, dist, moma_flux = sim.run_MOMA(
    wild_flux=flux_dist,
    flux_constraints=knockout_constraints
)

# ROOM으로 예측
status, n_changed, room_flux = sim.run_ROOM(
    wild_flux=flux_dist,
    flux_constraints=knockout_constraints,
    delta=0.03
)

print(f"ROOM: {n_changed}개 반응 변화")
