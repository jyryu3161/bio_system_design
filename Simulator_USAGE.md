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

첨부하신 README를 기반으로 각 예제를 독립적으로 실행 가능한 개별 파일로 수정하겠습니다.

```python
# 1_simple_fba.py
"""
기본 FBA 분석 예제
Flux Balance Analysis를 수행하여 최적 성장 속도를 예측합니다.
"""

from Simulator import Simulator

def main():
    # 시뮬레이터 초기화
    sim = Simulator()
    
    # 대사 모델 로드 (SBML 형식)
    print("모델 로딩 중...")
    sim.read_model('iML1515.xml')
    
    # FBA 실행
    print("\nFBA 실행 중...")
    status, objective_value, flux_distribution = sim.run_FBA()
    
    if status == 'optimal':
        print(f"\n=== FBA 결과 ===")
        print(f"최적화 상태: {status}")
        print(f"최적 성장 속도: {objective_value:.4f} hr⁻¹")
        print(f"활성 반응 수: {sum(1 for v in flux_distribution.values() if abs(v) > 1e-6)}")
        print(f"전체 반응 수: {len(flux_distribution)}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()
```

```python
# 2_cobrapy_model.py
"""
COBRApy 모델 사용 예제
COBRApy 라이브러리의 모델을 직접 로드하여 FBA를 수행합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # E. coli iML1515 모델 로드
    print("COBRApy에서 모델 로딩 중...")
    model = cobra.io.load_model("iML1515")
    print(f"모델명: {model.id}")
    print(f"반응 수: {len(model.reactions)}")
    print(f"대사물질 수: {len(model.metabolites)}")
    print(f"유전자 수: {len(model.genes)}")
    
    # 시뮬레이터에 모델 로드
    sim = Simulator()
    sim.load_cobra_model(model)
    
    # FBA 실행
    print("\nFBA 실행 중...")
    status, growth_rate, fluxes = sim.run_FBA()
    
    if status == 'optimal':
        print(f"\n=== FBA 결과 ===")
        print(f"최적화 상태: {status}")
        print(f"성장 속도: {growth_rate:.4f} hr⁻¹")
        
        # 주요 교환 반응 플럭스 출력
        print(f"\n주요 교환 반응:")
        exchange_reactions = ['EX_glc__D_e', 'EX_o2_e', 'EX_co2_e', 'EX_ac_e']
        for rxn_id in exchange_reactions:
            if rxn_id in fluxes:
                print(f"  {rxn_id}: {fluxes[rxn_id]:.4f}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()
```

```python
# 3_pfba.py
"""
Parsimonious FBA (pFBA) 예제
최적 성장을 유지하면서 총 플럭스를 최소화합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 표준 FBA 실행
    print("\n=== 표준 FBA 실행 ===")
    status_fba, growth_fba, fluxes_fba = sim.run_FBA()
    
    if status_fba == 'optimal':
        total_flux_fba = sum(abs(v) for v in fluxes_fba.values())
        active_rxns_fba = sum(1 for v in fluxes_fba.values() if abs(v) > 1e-6)
        
        print(f"성장 속도: {growth_fba:.4f} hr⁻¹")
        print(f"총 플럭스 합계: {total_flux_fba:.4f}")
        print(f"활성 반응 수: {active_rxns_fba}")
    
    # Parsimonious FBA 실행
    print("\n=== Parsimonious FBA (pFBA) 실행 ===")
    status_pfba, total_flux_pfba, fluxes_pfba = sim.run_FBA(
        internal_flux_minimization=True
    )
    
    if status_pfba == 'optimal':
        growth_pfba = fluxes_pfba[sim.objective]
        active_rxns_pfba = sum(1 for v in fluxes_pfba.values() if abs(v) > 1e-6)
        
        print(f"성장 속도: {growth_pfba:.4f} hr⁻¹")
        print(f"총 플럭스 합계: {total_flux_pfba:.4f}")
        print(f"활성 반응 수: {active_rxns_pfba}")
        
        # 비교
        print(f"\n=== FBA vs pFBA 비교 ===")
        print(f"총 플럭스 감소: {total_flux_fba - total_flux_pfba:.4f} ({(1 - total_flux_pfba/total_flux_fba)*100:.1f}%)")
        print(f"활성 반응 감소: {active_rxns_fba - active_rxns_pfba} 개")
    else:
        print(f"최적화 실패: {status_pfba}")

if __name__ == "__main__":
    main()
```

```python
# 4_moma_knockout.py
"""
MOMA를 사용한 유전자 Knockout 시뮬레이션
유전자 knockout 후 대사 플럭스 변화를 최소화하는 방식으로 표현형을 예측합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 야생형 (참조) 플럭스 분포 획득
    print("\n=== 야생형 FBA 실행 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    
    # PGI 유전자 knockout 시뮬레이션
    print("\n=== PGI 유전자 Knockout (MOMA) ===")
    knockout_constraints = {'PGI': (0, 0)}  # PGI 반응 차단
    
    # MOMA 실행하여 knockout 표현형 예측
    status, distance, moma_fluxes = sim.run_MOMA(
        wild_flux=wt_fluxes,
        flux_constraints=knockout_constraints
    )
    
    if status == 'optimal':
        knockout_growth = moma_fluxes[sim.objective]
        print(f"최적화 상태: {status}")
        print(f"MOMA 예측 성장 속도: {knockout_growth:.4f} hr⁻¹")
        print(f"성장 감소: {(1 - knockout_growth/wt_growth)*100:.1f}%")
        print(f"대사 거리 (L1 norm): {distance:.4f}")
        
        # 플럭스 변화가 큰 반응 찾기
        flux_changes = {
            rxn: abs(moma_fluxes[rxn] - wt_fluxes[rxn]) 
            for rxn in wt_fluxes.keys()
        }
        top_changes = sorted(flux_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n가장 많이 변화된 상위 5개 반응:")
        for rxn, change in top_changes:
            print(f"  {rxn}: {wt_fluxes[rxn]:.4f} → {moma_fluxes[rxn]:.4f} (변화: {change:.4f})")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()
```

```python
# 5_room_knockout.py
"""
ROOM을 사용한 유전자 Knockout 시뮬레이션
유의미하게 변화된 반응의 수를 최소화하여 knockout 표현형을 예측합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 야생형 플럭스 분포 획득
    print("\n=== 야생형 FBA 실행 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    
    # PGI 유전자 knockout 시뮬레이션
    print("\n=== PGI 유전자 Knockout (ROOM) ===")
    knockout_constraints = {'PGI': (0, 0)}
    
    # ROOM 실행하여 knockout 표현형 예측
    status, n_changed, room_fluxes = sim.run_ROOM(
        wild_flux=wt_fluxes,
        flux_constraints=knockout_constraints,
        delta=0.03,  # 3% 상대 허용오차
        epsilon=0.001  # 0에 가까운 플럭스의 절대 허용오차
    )
    
    if status == 'optimal':
        knockout_growth = room_fluxes[sim.objective]
        print(f"최적화 상태: {status}")
        print(f"ROOM 예측 성장 속도: {knockout_growth:.4f} hr⁻¹")
        print(f"성장 감소: {(1 - knockout_growth/wt_growth)*100:.1f}%")
        print(f"유의미하게 변화된 반응 수: {int(n_changed)}")
        
        # 변화된 반응 찾기 (delta와 epsilon 기준 적용)
        changed_reactions = []
        for rxn in wt_fluxes.keys():
            wt_val = abs(wt_fluxes[rxn])
            room_val = abs(room_fluxes[rxn])
            
            # ROOM의 변화 기준 적용
            threshold = max(0.03 * wt_val, 0.001)
            if abs(room_val - wt_val) > threshold:
                changed_reactions.append((rxn, wt_fluxes[rxn], room_fluxes[rxn]))
        
        print(f"\n유의미하게 변화된 반응 (상위 5개):")
        for rxn, wt_val, ko_val in changed_reactions[:5]:
            print(f"  {rxn}: {wt_val:.4f} → {ko_val:.4f}")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()
```

```python
# 6_fseof_analysis.py
"""
FSEOF를 사용한 생산 Envelope 분석
제약된 성장 조건에서 목표 대사물질의 생산 범위를 탐색합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 먼저 최대 성장 속도 확인
    print("\n=== 야생형 FBA 실행 ===")
    _, max_growth, _ = sim.run_FBA()
    print(f"최대 성장 속도: {max_growth:.4f} hr⁻¹")
    
    # 숙신산 생산을 위한 FSEOF 분석
    target_reaction = 'EX_succ_e'  # 숙신산 분비 반응
    objective_fraction = 0.9  # 최대 성장의 90%
    
    print(f"\n=== FSEOF 분석: {target_reaction} ===")
    print(f"성장 제약: 최대 성장의 {objective_fraction*100:.0f}%")
    
    # FSEOF 실행
    status, min_flux, max_flux, flux_range = sim.run_FSEOF(
        target_reaction=target_reaction,
        objective_fraction=objective_fraction
    )
    
    if status == 'optimal':
        print(f"\n최적화 상태: {status}")
        print(f"제약된 성장 속도: {max_growth * objective_fraction:.4f} hr⁻¹")
        print(f"\n숙신산 생산 범위:")
        print(f"  최소 생산: {min_flux:.4f} mmol/gDW/hr")
        print(f"  최대 생산: {max_flux:.4f} mmol/gDW/hr")
        print(f"  생산 범위: {flux_range:.4f} mmol/gDW/hr")
        
        # 최대 생산 시 성장률 확인
        print(f"\n최대 생산 달성 가능 여부:")
        if max_flux > 1e-6:
            print(f"  성장을 {objective_fraction*100:.0f}% 유지하면서 최대 {max_flux:.4f} mmol/gDW/hr 생산 가능")
        else:
            print(f"  해당 성장 조건에서는 숙신산 생산 불가")
    else:
        print(f"최적화 실패: {status}")
        print("해당 조건에서는 생산이 불가능합니다.")

if __name__ == "__main__":
    main()
```

```python
# 7_fvseof_analysis.py
"""
FVSEOF를 사용한 대사 유연성 분석
여러 생산 수준에서 대사 네트워크의 유연성을 종합적으로 분석합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 최대 성장 속도 확인
    print("\n=== 야생형 FBA 실행 ===")
    _, max_growth, _ = sim.run_FBA()
    print(f"최대 성장 속도: {max_growth:.4f} hr⁻¹")
    
    # 숙신산 생산에 대한 FVSEOF 분석
    target_reaction = 'EX_succ_e'
    objective_fraction = 0.9
    num_points = 10
    
    print(f"\n=== FVSEOF 분석: {target_reaction} ===")
    print(f"성장 제약: 최대 성장의 {objective_fraction*100:.0f}%")
    print(f"분석 포인트: {num_points}개")
    
    # FVSEOF 실행
    status, fvseof_results = sim.run_FVSEOF(
        target_reaction=target_reaction,
        objective_fraction=objective_fraction,
        num_points=num_points
    )
    
    if status == 'optimal':
        print(f"\n최적화 상태: {status}")
        print(f"분석된 포인트 수: {len(fvseof_results['target_fluxes'])}")
        
        print(f"\n{'포인트':<8} {'생산량':<12} {'최소 성장':<12} {'최대 성장':<12} {'성장 범위':<12}")
        print("=" * 60)
        
        # 각 포인트에서 대사 유연성 분석
        for i, target_flux in enumerate(fvseof_results['target_fluxes']):
            obj_flux = fvseof_results['objective_fluxes'][i]
            min_growth = obj_flux['minimum']
            max_growth = obj_flux['maximum']
            growth_range = max_growth - min_growth
            
            print(f"{i+1:<8} {target_flux:<12.4f} {min_growth:<12.4f} {max_growth:<12.4f} {growth_range:<12.4f}")
        
        # 요약 통계
        print(f"\n=== 요약 ===")
        print(f"최소 생산량: {min(fvseof_results['target_fluxes']):.4f}")
        print(f"최대 생산량: {max(fvseof_results['target_fluxes']):.4f}")
        
        growth_ranges = [
            fvseof_results['objective_fluxes'][i]['maximum'] - 
            fvseof_results['objective_fluxes'][i]['minimum']
            for i in range(len(fvseof_results['target_fluxes']))
        ]
        avg_flexibility = sum(growth_ranges) / len(growth_ranges)
        print(f"평균 성장 유연성: {avg_flexibility:.4f} hr⁻¹")
    else:
        print(f"최적화 실패: {status}")

if __name__ == "__main__":
    main()
```

```python
# 8_custom_objective.py
"""
사용자 정의 목적 함수 예제
특정 대사물질의 생산을 최대화하거나 다른 목적 함수를 사용합니다.
"""

from Simulator import Simulator
import cobra

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 기본 성장 최대화 (바이오매스)
    print("\n=== 기본 목적 함수: 성장 최대화 ===")
    status, growth_rate, fluxes = sim.run_FBA()
    print(f"최대 성장 속도: {growth_rate:.4f} hr⁻¹")
    print(f"숙신산 생산: {fluxes.get('EX_succ_e', 0):.4f} mmol/gDW/hr")
    
    # 숙신산 생산 최대화
    print("\n=== 사용자 정의 목적 함수: 숙신산 생산 최대화 ===")
    status, production_rate, fluxes = sim.run_FBA(
        new_objective='EX_succ_e',
        mode='max'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]  # 원래 목적 함수 (바이오매스) 값
        print(f"최대 숙신산 생산: {production_rate:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")
    
    # 아세트산 생산 최대화
    print("\n=== 사용자 정의 목적 함수: 아세트산 생산 최대화 ===")
    status, production_rate, fluxes = sim.run_FBA(
        new_objective='EX_ac_e',
        mode='max'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]
        print(f"최대 아세트산 생산: {production_rate:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")
    
    # ATP 유지 비용 최소화
    print("\n=== 사용자 정의 목적 함수: ATP 유지 비용 최소화 ===")
    status, atp_cost, fluxes = sim.run_FBA(
        new_objective='ATPM',
        mode='min'
    )
    
    if status == 'optimal':
        growth_rate = fluxes[sim.objective]
        print(f"최소 ATP 유지 비용: {atp_cost:.4f} mmol/gDW/hr")
        print(f"해당 조건에서 성장 속도: {growth_rate:.4f} hr⁻¹")

if __name__ == "__main__":
    main()
```

```python
# 10_multiple_gene_knockout.py
"""
다중 유전자 Knockout 분석
여러 유전자의 knockout 효과를 MOMA와 ROOM으로 비교 분석합니다.
"""

from Simulator import Simulator
import cobra
import pandas as pd

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 야생형 플럭스 분포 획득
    print("\n=== 야생형 FBA 실행 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    
    # 테스트할 유전자 목록 (해당계 경로의 주요 효소)
    genes_to_test = ['PGI', 'PFK', 'FBA', 'TPI', 'GAPD']
    
    print(f"\n=== {len(genes_to_test)}개 유전자 Knockout 분석 ===")
    results = []
    
    for gene in genes_to_test:
        print(f"\n분석 중: {gene}")
        
        # Knockout 제약 생성
        knockout = {gene: (0, 0)}
        
        # MOMA로 예측
        status_moma, dist, moma_flux = sim.run_MOMA(wt_fluxes, knockout)
        moma_growth = moma_flux[sim.objective] if status_moma == 'optimal' else 0
        
        # ROOM으로 예측
        status_room, n_changed, room_flux = sim.run_ROOM(wt_fluxes, knockout)
        room_growth = room_flux[sim.objective] if status_room == 'optimal' else 0
        
        results.append({
            '유전자': gene,
            '야생형 성장': wt_growth,
            'MOMA 성장': moma_growth,
            'ROOM 성장': room_growth,
            'MOMA 감소율 (%)': (1 - moma_growth/wt_growth) * 100 if wt_growth > 0 else 100,
            'ROOM 감소율 (%)': (1 - room_growth/wt_growth) * 100 if wt_growth > 0 else 100,
            '변화된 반응 수': int(n_changed) if status_room == 'optimal' else 'N/A',
            'MOMA 상태': status_moma,
            'ROOM 상태': status_room
        })
    
    # 결과를 DataFrame으로 변환하여 표시
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("=== Knockout 분석 결과 요약 ===")
    print("="*100)
    print(df[['유전자', '야생형 성장', 'MOMA 성장', 'ROOM 성장', 
              'MOMA 감소율 (%)', 'ROOM 감소율 (%)', '변화된 반응 수']].to_string(index=False))
    
    # 필수 유전자 판별
    print("\n=== 필수 유전자 분석 ===")
    essential_genes = df[df['MOMA 감소율 (%)'] > 99].copy()
    if len(essential_genes) > 0:
        print(f"필수 유전자 (성장 99% 이상 감소): {', '.join(essential_genes['유전자'].tolist())}")
    else:
        print("테스트한 유전자 중 필수 유전자 없음")
    
    # MOMA vs ROOM 비교
    print("\n=== MOMA vs ROOM 예측 비교 ===")
    avg_diff = abs(df['MOMA 감소율 (%)'] - df['ROOM 감소율 (%)']).mean()
    print(f"평균 예측 차이: {avg_diff:.2f}%")
    print(f"평균 변화된 반응 수: {df[df['변화된 반응 수'] != 'N/A']['변화된 반응 수'].mean():.1f}개")

if __name__ == "__main__":
    main()
```

```python
# 11_knockout_production_analysis.py
"""
Knockout 시 성장과 목표 생산물 분석
유전자 knockout이 성장과 특정 대사물질 생산에 미치는 영향을 분석합니다.
"""

from Simulator import Simulator
import cobra
import pandas as pd

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 목표 생산물 설정
    target_product = 'EX_succ_e'  # 숙신산
    
    # 야생형 분석
    print("\n=== 야생형 분석 ===")
    _, wt_growth, wt_fluxes = sim.run_FBA()
    wt_production = wt_fluxes.get(target_product, 0)
    
    print(f"야생형 성장 속도: {wt_growth:.4f} hr⁻¹")
    print(f"야생형 {target_product} 생산: {wt_production:.4f} mmol/gDW/hr")
    
    # 야생형에서 목표 생산물 최대 생산
    _, wt_max_prod, wt_max_fluxes = sim.run_FBA(new_objective=target_product, mode='max')
    wt_max_growth = wt_max_fluxes[sim.objective]
    print(f"야생형 최대 {target_product} 생산: {wt_max_prod:.4f} mmol/gDW/hr (성장: {wt_max_growth:.4f})")
    
    # Knockout 테스트할 유전자
    genes_to_test = ['PGI', 'PFK', 'FBA', 'PYK']
    
    print(f"\n=== {len(genes_to_test)}개 유전자 Knockout 후 성장 및 생산 분석 ===")
    results = []
    
    for gene in genes_to_test:
        print(f"\n분석 중: {gene} knockout")
        
        knockout = {gene: (0, 0)}
        
        # MOMA로 knockout 예측
        status_moma, _, moma_fluxes = sim.run_MOMA(wt_fluxes, knockout)
        
        if status_moma == 'optimal':
            # Knockout 후 성장 및 생산
            moma_growth = moma_fluxes[sim.objective]
            moma_production = moma_fluxes.get(target_product, 0)
            
            # Knockout 후 최대 생산 분석
            # 새로운 시뮬레이터로 knockout 조건 설정
            sim_ko = Simulator()
            sim_ko.load_cobra_model(model)
            _, ko_max_prod, ko_max_fluxes = sim_ko.run_FBA(
                new_objective=target_product,
                flux_constraints=knockout,
                mode='max'
            )
            ko_max_growth = ko_max_fluxes[sim_ko.objective]
            
            results.append({
                '유전자': gene,
                '성장 (WT)': wt_growth,
                '성장 (KO)': moma_growth,
                '성장 감소율 (%)': (1 - moma_growth/wt_growth) * 100,
                '생산 (WT)': wt_production,
                '생산 (KO)': moma_production,
                '생산 변화율 (%)': ((moma_production - wt_production) / abs(wt_production) * 100) if abs(wt_production) > 1e-6 else 0,
                '최대 생산 (WT)': wt_max_prod,
                '최대 생산 (KO)': ko_max_prod,
                '최대 생산 시 성장 (KO)': ko_max_growth,
                '상태': 'Success'
            })
        else:
            results.append({
                '유전자': gene,
                '성장 (WT)': wt_growth,
                '성장 (KO)': 0,
                '성장 감소율 (%)': 100,
                '생산 (WT)': wt_production,
                '생산 (KO)': 0,
                '생산 변화율 (%)': -100,
                '최대 생산 (WT)': wt_max_prod,
                '최대 생산 (KO)': 0,
                '최대 생산 시 성장 (KO)': 0,
                '상태': 'Infeasible'
            })
    
    # 결과 표시
    df = pd.DataFrame(results)
    
    print("\n" + "="*120)
    print("=== Knockout 후 성장 분석 ===")
    print("="*120)
    print(df[['유전자', '성장 (WT)', '성장 (KO)', '성장 감소율 (%)', '상태']].to_string(index=False))
    
    print("\n" + "="*120)
    print(f"=== Knockout 후 {target_product} 생산 분석 ===")
    print("="*120)
    print(df[['유전자', '생산 (WT)', '생산 (KO)', '생산 변화율 (%)']].to_string(index=False))
    
    print("\n" + "="*120)
    print(f"=== Knockout 후 최대 {target_product} 생산 능력 ===")
    print("="*120)
    print(df[['유전자', '최대 생산 (WT)', '최대 생산 (KO)', '최대 생산 시 성장 (KO)']].to_string(index=False))
    
    # 생산 증가 전략 제안
    print("\n=== 생산 최적화 전략 제안 ===")
    viable_knockouts = df[(df['상태'] == 'Success') & (df['성장 (KO)'] > 0.1 * wt_growth)]
    
    if len(viable_knockouts) > 0:
        # 최대 생산이 증가한 경우
        improved = viable_knockouts[viable_knockouts['최대 생산 (KO)'] > wt_max_prod * 1.01]
        if len(improved) > 0:
            print(f"\n최대 생산 증가 knockout 후보:")
            for _, row in improved.iterrows():
                increase = (row['최대 생산 (KO)'] / wt_max_prod - 1) * 100
                print(f"  {row['유전자']}: 최대 생산 {increase:.1f}% 증가 "
                      f"(성장 {row['성장 감소율 (%)']:.1f}% 감소)")
        
        # 성장은 유지하면서 생산이 가능한 경우
        maintained = viable_knockouts[viable_knockouts['성장 (KO)'] > 0.9 * wt_growth]
        if len(maintained) > 0:
            print(f"\n성장 유지 knockout 후보 (10% 이내 감소):")
            for _, row in maintained.iterrows():
                print(f"  {row['유전자']}: 성장 {row['성장 감소율 (%)']:.1f}% 감소, "
                      f"최대 생산 {row['최대 생산 (KO)']:.4f}")
    else:
        print("성장을 크게 유지하면서 생산을 개선할 수 있는 knockout 후보가 없습니다.")

if __name__ == "__main__":
    main()
```

각 예제 파일은:
1. 독립적으로 실행 가능
2. 필요한 모든 import 포함
3. 명확한 주석과 설명
4. 결과를 보기 쉽게 출력

실행 방법:
```bash
cd simulator_examples
python 1_simple_fba.py
python 2_cobrapy_model.py
# ... 등등
```
