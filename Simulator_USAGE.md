# Metabolic Model Simulator 사용 가이드

Python 기반의 제약 기반 대사 모델링 시뮬레이터로, Flux Balance Analysis (FBA), Minimization of Metabolic Adjustment (MOMA), Regulatory On/Off Minimization (ROOM) 방법을 구현합니다.

## 개요

이 시뮬레이터는 genome-scale 대사 모델 분석을 위한 다양한 제약 기반 방법론을 제공합니다:

- **FBA (Flux Balance Analysis)**: 목적 함수를 최적화하여 대사 플럭스 분포를 예측
- **pFBA (Parsimonious FBA)**: 최적 성장을 유지하면서 총 플럭스를 최소화
- **MOMA (Minimization of Metabolic Adjustment)**: 유전적 변이 후 플럭스 변화를 최소화하여 대사 행동을 예측
- **ROOM (Regulatory On/Off Minimization)**: 유의미하게 변화된 반응의 수를 최소화하여 knockout 표현형을 예측
- **FSEOF (Flux Scanning based on Enforced Objective Flux)**: 제약된 성장 조건에서 목표 반응의 가능한 플럭스 범위를 탐색
- **FVSEOF (Flux Variability Scanning based on Enforced Objective Flux)**: 목표 플럭스의 여러 수준에서 대사 유연성을 종합적으로 분석

## 주요 기능

- ✅ SBML 파일 또는 COBRApy 모델 객체에서 모델 로드
- ✅ 표준 및 parsimonious FBA 수행
- ✅ MOMA 또는 ROOM을 사용한 유전자 knockout 시뮬레이션
- ✅ FSEOF/FVSEOF를 통한 생산 envelope 및 대사 유연성 분석
- ✅ 유연한 플럭스 제약 및 사용자 정의 목적 함수
- ✅ 포괄적인 문서 및 예제

## 설치

### Step 1: 저장소 클론

먼저 GitHub에서 프로젝트 저장소를 클론합니다.

```bash
git clone https://github.com/jyryu3161/bio_system_design.git
cd bio_system_design
```

### Step 2: Pixi로 환경 설치

Pixi를 사용하여 필요한 모든 의존성을 자동으로 설치합니다.

```bash
pixi install
```

이 명령은 `pixi.toml` 파일에 정의된 모든 패키지를 설치합니다:
- Python 3.10
- COBRApy (대사 모델링 라이브러리)
- SCIP Optimization Suite (오픈소스 solver)
- 과학 계산 라이브러리 (NumPy, SciPy, Pandas)
- 시각화 도구 (Matplotlib, Seaborn)
- Jupyter Notebook/Lab
- 기타 대사 모델링 도구 (CarveMe, MEMOTE)

### Step 3: Pixi 환경 활성화

설치가 완료되면 Pixi 환경을 활성화합니다.

```bash
pixi shell
```

환경이 활성화되면 프롬프트가 변경되어 현재 `gem` 환경에 있음을 표시합니다.

### Step 4: IBM CPLEX 설치 (선택사항, 권장)

SCIP solver도 사용 가능하지만, 더 나은 성능을 위해 IBM CPLEX를 설치할 수 있습니다.

#### CPLEX 학술용 무료 라이선스

1. **IBM Academic Initiative에서 무료 학술 라이선스 등록**: [IBM Academic Initiative](https://www.ibm.com/academic/home)

2. **CPLEX 다운로드 및 설치**: 

```bash
# 다운로드한 설치 파일에 실행 권한 부여
chmod +x cplex_studio2211.linux-x86-64.bin

# CPLEX 설치 (설치 경로를 기억해두세요, 예: /home/biosys/solver)
./cplex_studio2211.linux_x86_64.bin
```

3. **CPLEX Python 모듈 설치**:

```bash
# Pixi 환경 활성화 상태에서
cd /home/biosys/solver/cplex/python/3.10/x86-64_linux  # CPLEX 설치 경로에 맞게 수정

# CPLEX Python 모듈 설치
pixi run python setup.py install
```

4. **설치 확인**:

```python
python
>>> import cplex
>>> import cobra
>>> print(f"COBRApy version: {cobra.__version__}")
>>> print(f"CPLEX version: {cplex.__version__}")
>>> exit()
```

### Step 5: 설치 확인

환경이 제대로 설정되었는지 확인합니다.

```python
import cobra
import numpy as np
import pandas as pd

print(f"COBRApy version: {cobra.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Solver 확인
print(f"Available solvers: {cobra.Configuration().solver}")
```

예상 출력:
```
COBRApy version: 0.30.0
NumPy version: 1.24.x
Pandas version: 2.x.x
Available solvers: cplex (또는 scip)
```

## 빠른 시작

```
cd simulator_examples
```

### 1. 기본 FBA 분석

```python
from simulator import Simulator

# 시뮬레이터 초기화
sim = Simulator()

# 대사 모델 로드 (SBML 형식)
sim.read_model('iML1515.xml')

# FBA 실행
status, objective_value, flux_distribution = sim.run_FBA()

if status == 'optimal':
    print(f"최적 성장 속도: {objective_value:.4f}")
    print(f"활성 반응 수: {sum(1 for v in flux_distribution.values() if abs(v) > 1e-6)}")
```

### 2. COBRApy 모델 사용

```python
from simulator import Simulator
import cobra

# E. coli iML1515 모델 로드
sim = Simulator()
model = cobra.io.load_model("iML1515")
sim.load_cobra_model(model)

# FBA 실행
status, growth_rate, fluxes = sim.run_FBA()
print(f"성장 속도: {growth_rate:.4f} hr⁻¹")
```

### 3. Parsimonious FBA (pFBA)

```python
# 최적 성장을 유지하면서 총 플럭스 최소화
status, total_flux, fluxes = sim.run_FBA(internal_flux_minimization=True)

if status == 'optimal':
    print(f"총 플럭스 합계: {total_flux:.4f}")
```

### 4. MOMA를 사용한 유전자 Knockout 시뮬레이션

```python
from simulator import Simulator
import cobra

# 모델 로드 및 야생형 플럭스 분포 획득
sim = Simulator()
model = cobra.io.load_model("iML1515")
sim.load_cobra_model(model)

# 야생형 (참조) 플럭스 분포 획득
_, wt_growth, wt_fluxes = sim.run_FBA()
print(f"야생형 성장 속도: {wt_growth:.4f}")

# PGI 유전자 knockout 시뮬레이션
knockout_constraints = {'PGI': (0, 0)}  # PGI 반응 차단

# MOMA 실행하여 knockout 표현형 예측
status, distance, moma_fluxes = sim.run_MOMA(
    wild_flux=wt_fluxes,
    flux_constraints=knockout_constraints
)

if status == 'optimal':
    knockout_growth = moma_fluxes[sim.objective]
    print(f"MOMA 예측 성장 속도: {knockout_growth:.4f}")
    print(f"성장 감소: {(1 - knockout_growth/wt_growth)*100:.1f}%")
    print(f"대사 거리: {distance:.4f}")
```

### 5. ROOM을 사용한 유전자 Knockout 시뮬레이션

```python
# ROOM 실행하여 knockout 표현형 예측
status, n_changed, room_fluxes = sim.run_ROOM(
    wild_flux=wt_fluxes,
    flux_constraints=knockout_constraints,
    delta=0.03,  # 3% 상대 허용오차
    epsilon=0.001  # 0에 가까운 플럭스의 절대 허용오차
)

if status == 'optimal':
    knockout_growth = room_fluxes[sim.objective]
    print(f"ROOM 예측 성장 속도: {knockout_growth:.4f}")
    print(f"유의미하게 변화된 반응 수: {int(n_changed)}")
```

### 6. FSEOF를 사용한 생산 Envelope 분석

```python
# 최적 성장을 유지하면서 목표 대사물질의 생산 범위 탐색
target_reaction = 'EX_succ_e'  # 숙신산 분비 반응

# FSEOF 실행
status, min_flux, max_flux, flux_range = sim.run_FSEOF(
    target_reaction=target_reaction,
    objective_fraction=0.9  # 최대 성장의 90% 유지
)

if status == 'optimal':
    print(f"90% 성장에서 숙신산 생산 범위:")
    print(f"  최소: {min_flux:.4f}")
    print(f"  최대: {max_flux:.4f}")
    print(f"  범위: {max_flux - min_flux:.4f}")
```

### 7. FVSEOF를 사용한 대사 유연성 분석

```python
# 여러 생산 수준에서 플럭스 가변성 분석
status, fvseof_results = sim.run_FVSEOF(
    target_reaction='EX_succ_e',
    objective_fraction=0.9,
    num_points=10  # 10개의 생산 수준에서 분석
)

if status == 'optimal':
    print(f"분석된 포인트 수: {len(fvseof_results['target_fluxes'])}")
    
    # 각 포인트에서 대사 유연성 분석
    for i, target_flux in enumerate(fvseof_results['target_fluxes']):
        obj_flux = fvseof_results['objective_fluxes'][i]
        print(f"생산 수준 {i+1}: {target_flux:.4f}")
        print(f"  성장 범위: {obj_flux['minimum']:.4f} - {obj_flux['maximum']:.4f}")
```

### 8. 사용자 정의 목적 함수

```python
# 특정 대사물질의 생산 최대화
status, production_rate, fluxes = sim.run_FBA(
    new_objective='EX_succ_e',  # 숙신산 분비 반응
    mode='max'
)

print(f"최대 숙신산 생산: {production_rate:.4f}")
```

### 9. Flux Variability Analysis (FVA)

```python
# 최적 성장을 유지하면서 각 반응의 플럭스 범위 확인
_, optimal_growth, _ = sim.run_FBA()

# FVA 실행
fva_result = sim.run_FVA(
    fraction_of_optimum=0.99,  # 최적값의 99% 유지
    reactions_to_analyze=sim.model_reactions[:50]  # 처음 50개 반응 분석
)

if fva_result['status'] == 'optimal':
    print(f"FVA 완료: {len(fva_result['fva_data'])}개 반응 분석")
    
    # 필수 반응 찾기 (플럭스를 반드시 가져야 하는 반응)
    essential_reactions = [
        r for r, data in fva_result['fva_data'].items()
        if data['minimum'] > 1e-6 or data['maximum'] < -1e-6
    ]
    print(f"필수 반응 수: {len(essential_reactions)}")
```

### 10. 다중 유전자 Knockout 분석

```python
import pandas as pd

# 여러 유전자 knockout 테스트
genes_to_test = ['PGI', 'PFK', 'FBA', 'TPI', 'GAPD']
results = []

for gene in genes_to_test:
    # Knockout 제약 생성
    knockout = {gene: (0, 0)}
    
    # MOMA로 예측
    status, dist, moma_flux = sim.run_MOMA(wt_fluxes, knockout)
    moma_growth = moma_flux[sim.objective] if status == 'optimal' else 0
    
    # ROOM으로 예측
    status, n_changed, room_flux = sim.run_ROOM(wt_fluxes, knockout)
    room_growth = room_flux[sim.objective] if status == 'optimal' else 0
    
    results.append({
        '유전자': gene,
        '야생형 성장': wt_growth,
        'MOMA 성장': moma_growth,
        'ROOM 성장': room_growth,
        'MOMA 감소율 (%)': (1 - moma_growth/wt_growth) * 100,
        'ROOM 감소율 (%)': (1 - room_growth/wt_growth) * 100,
        '변화된 반응 수': n_changed if status == 'optimal' else 'N/A'
    })

# 결과 표시
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 11. 야생형과 Knockout 플럭스 분포 비교

```python
import numpy as np

# Knockout 예측 수행
knockout = {'PGI': (0, 0)}
_, _, moma_flux = sim.run_MOMA(wt_fluxes, knockout)

# 플럭스 분포 비교
reactions = list(wt_fluxes.keys())
wt_values = [wt_fluxes[r] for r in reactions]
ko_values = [moma_flux[r] for r in reactions]

# 플럭스 차이 계산
differences = [abs(wt_values[i] - ko_values[i]) for i in range(len(reactions))]

# 가장 많이 변화된 상위 10개 반응 찾기
top_changed = sorted(zip(reactions, differences), key=lambda x: x[1], reverse=True)[:10]

print("가장 많이 변화된 상위 10개 반응:")
for reaction, diff in top_changed:
    wt_val = wt_fluxes[reaction]
    ko_val = moma_flux[reaction]
    print(f"{reaction:20s} WT: {wt_val:8.4f}  KO: {ko_val:8.4f}  차이: {diff:8.4f}")
```

### 12. 생산 Envelope 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

# 여러 성장 수준에서 FSEOF 실행
target_reaction = 'EX_succ_e'
fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
envelope_data = []

for frac in fractions:
    status, min_f, max_f, _ = sim.run_FSEOF(
        target_reaction=target_reaction,
        objective_fraction=frac
    )
    if status == 'optimal':
        envelope_data.append({
            'fraction': frac,
            'min': min_f,
            'max': max_f
        })

# 시각화
fracs = [e['fraction'] for e in envelope_data]
mins = [e['min'] for e in envelope_data]
maxs = [e['max'] for e in envelope_data]

plt.figure(figsize=(10, 6))
plt.fill_between(fracs, mins, maxs, alpha=0.3, color='blue')
plt.plot(fracs, mins, 'b-', marker='o', label='최소')
plt.plot(fracs, maxs, 'r-', marker='s', label='최대')
plt.xlabel('목적 함수 비율')
plt.ylabel(f'{target_reaction} 플럭스')
plt.title('FSEOF: 생산 Envelope')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('production_envelope.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 사용자 정의 플럭스 제약

```python
# 포도당 흡수율 설정
constraints = {
    'EX_glc__D_e': (-10, -10),  # 포도당 흡수를 10 mmol/gDW/hr로 고정
    'EX_o2_e': (-20, 0),        # 산소 흡수 제한
    'ATPM': (8.39, 8.39)        # ATP 유지 비용 고정
}

status, growth, fluxes = sim.run_FBA(flux_constraints=constraints)
```

## 방법론 비교

| 방법 | 최적화 유형 | 사용 사례 | 계산 시간 |
|------|------------|----------|----------|
| **FBA** | 선형 계획법 (LP) | 최적 플럭스 분포 예측 | 빠름 (초) |
| **pFBA** | LP (2단계) | 보다 현실적인 플럭스 분포 | 빠름 (초) |
| **MOMA** | 선형 계획법 (L1 norm) | 즉각적인 knockout 반응 예측 | 중간 (초) |
| **ROOM** | 혼합 정수 LP (MILP) | 조절된 knockout 반응 예측 | 느림 (분) |
| **FSEOF** | LP (반복) | 생산 envelope 매핑 | 중간 (초~분) |
| **FVSEOF** | LP (다중 반복) | 대사 유연성 종합 분석 | 느림 (분~시간) |

### 각 방법을 사용하는 경우

- **FBA**: 다양한 조건에서 최대 성장 속도 또는 생산 예측
- **pFBA**: 무의미한 사이클 없이 보다 현실적인 플럭스 분포 획득
- **MOMA**: 유전자 knockout에 대한 즉각적인 반응 시뮬레이션 (조절 전)
- **ROOM**: 유전자 knockout에 대한 장기 적응 반응 시뮬레이션 (조절 후)
- **FSEOF**: 제약된 성장에서 생산 능력 평가
- **FVSEOF**: 생산 수준에 따른 대사 네트워크 유연성 분석

## API 참조

### Simulator 클래스

#### `__init__()`
새로운 Simulator 인스턴스를 초기화합니다.

#### `read_model(filename)`
SBML 파일에서 대사 모델을 로드합니다.

**매개변수:**
- `filename` (str): SBML 모델 파일 경로

**반환값:** 모델 구성 요소 튜플

#### `load_cobra_model(cobra_model)`
COBRApy 모델 객체를 로드합니다.

**매개변수:**
- `cobra_model` (cobra.Model): COBRApy 모델 객체

**반환값:** 모델 구성 요소 튜플

#### `run_FBA(new_objective='', flux_constraints={}, inf_flag=False, internal_flux_minimization=False, mode='max')`
Flux Balance Analysis를 수행합니다.

**매개변수:**
- `new_objective` (str): 목적 함수로 사용할 반응 ID (기본값: 모델의 목적 함수)
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `inf_flag` (bool): True일 경우 무한대 경계값 유지
- `internal_flux_minimization` (bool): True일 경우 pFBA 수행
- `mode` (str): 'max' 또는 'min'

**반환값:** (status, objective_value, flux_distribution)

#### `run_MOMA(wild_flux={}, flux_constraints={}, inf_flag=False)`
Minimization of Metabolic Adjustment를 수행합니다.

**매개변수:**
- `wild_flux` (dict): 참조 플럭스 분포
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `inf_flag` (bool): True일 경우 무한대 경계값 유지

**반환값:** (status, distance, flux_distribution)

#### `run_ROOM(wild_flux={}, flux_constraints={}, delta=0.03, epsilon=0.001, inf_flag=False)`
Regulatory On/Off Minimization을 수행합니다.

**매개변수:**
- `wild_flux` (dict): 참조 플럭스 분포
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `delta` (float): 플럭스 변화의 상대 허용오차 (기본값: 0.03)
- `epsilon` (float): 0에 가까운 플럭스의 절대 허용오차 (기본값: 0.001)
- `inf_flag` (bool): True일 경우 무한대 경계값 유지

**반환값:** (status, n_changed_reactions, flux_distribution)

#### `run_FSEOF(target_reaction, objective_fraction=1.0, flux_constraints={}, inf_flag=False)`
Flux Scanning based on Enforced Objective Flux를 수행합니다.

**매개변수:**
- `target_reaction` (str): 탐색할 반응 ID
- `objective_fraction` (float): 유지할 목적 함수의 비율 (0~1, 기본값: 1.0)
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `inf_flag` (bool): True일 경우 무한대 경계값 유지

**반환값:** (status, min_flux, max_flux, flux_range)

#### `run_FVSEOF(target_reaction, objective_fraction=1.0, num_points=10, flux_constraints={}, inf_flag=False)`
Flux Variability Scanning based on Enforced Objective Flux를 수행합니다.

**매개변수:**
- `target_reaction` (str): 탐색할 반응 ID
- `objective_fraction` (float): 유지할 목적 함수의 비율 (0~1, 기본값: 1.0)
- `num_points` (int): 샘플링할 플럭스 포인트 수 (기본값: 10)
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `inf_flag` (bool): True일 경우 무한대 경계값 유지

**반환값:** (status, fvseof_results)

#### `run_FVA(fraction_of_optimum=1.0, flux_constraints={}, inf_flag=False, reactions_to_analyze=None)`
Flux Variability Analysis를 수행합니다.

**매개변수:**
- `fraction_of_optimum` (float): 유지할 목적 함수의 비율 (0~1, 기본값: 1.0)
- `flux_constraints` (dict): 추가 플럭스 제약 {반응: (하한, 상한)}
- `inf_flag` (bool): True일 경우 무한대 경계값 유지
- `reactions_to_analyze` (list): 분석할 반응 ID 목록 (None일 경우 모든 반응)

**반환값:** {'status', 'fva_data', 'objective_value'}

## 인용

연구에서 이 시뮬레이터를 사용하는 경우 다음을 인용해 주세요:

### FBA
- Orth, J. D., Thiele, I., & Palsson, B. Ø. (2010). What is flux balance analysis? *Nature Biotechnology*, 28(3), 245-248.

### MOMA
- Segrè, D., Vitkup, D., & Church, G. M. (2002). Analysis of optimality in natural and perturbed metabolic networks. *Proceedings of the National Academy of Sciences*, 99(23), 15112-15117.

### ROOM
- Shlomi, T., Berkman, O., & Ruppin, E. (2005). Regulatory on/off minimization of metabolic flux changes after genetic perturbations. *Proceedings of the National Academy of Sciences*, 102(21), 7695-7700.

## 라이선스

MIT License
