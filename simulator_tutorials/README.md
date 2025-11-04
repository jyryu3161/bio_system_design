# Simulator Tutorials

이 폴더는 Metabolic Model Simulator의 실습 예제 코드와 튜토리얼을 포함합니다.

## 📁 폴더 구조

```
simulator_tutorials/
├── Simulator.py           # Simulator 클래스
├── iML1515.xml           # E. coli 대사 모델
├── README.md             # 이 파일
├── Tutorial_1_FBA_Basics.ipynb  # Jupyter 노트북 튜토리얼
│
├── 1_simple_fba.py       # 기본 FBA
├── 2_cobrapy_model.py    # COBRApy 모델 사용
├── 3_pfba.py             # Parsimonious FBA
├── 4_moma_knockout.py    # MOMA Knockout
├── 5_room_knockout.py    # ROOM Knockout
├── 6_fseof_analysis.py   # FSEOF 분석
├── 7_fvseof_analysis.py  # FVSEOF 분석
├── 8_custom_objective.py # 사용자 정의 목적 함수
└── 9_knockout_production_analysis.py  # Knockout 생산 분석
```

## 🚀 시작하기

### Python 스크립트 실행

```bash
# Pixi 환경 활성화 (필수)
cd /home/jyryu/lectures/bio_system_design
pixi shell

# simulator_tutorials 폴더로 이동
cd simulator_tutorials

# 예제 실행
python 1_simple_fba.py
python 2_cobrapy_model.py
# ... 등등
```

### Jupyter Notebook 실행

#### 방법 1: JupyterLab (권장)

```bash
# 프로젝트 루트에서
pixi run jupyter lab

# 브라우저에서 simulator_tutorials/Tutorial_1_FBA_Basics.ipynb 열기
```

#### 방법 2: Jupyter Notebook

```bash
# simulator_tutorials 폴더에서
cd simulator_tutorials
pixi run jupyter notebook

# 브라우저에서 Tutorial_1_FBA_Basics.ipynb 열기
```

## 📚 예제 목록

### 기본 예제

1. **1_simple_fba.py** - 기본 FBA 분석
   - SBML 파일에서 모델 로드
   - 최적 성장 속도 예측
   - 플럭스 분포 확인

2. **2_cobrapy_model.py** - COBRApy 모델 사용
   - COBRApy에서 직접 모델 로드
   - 주요 교환 반응 분석

3. **3_pfba.py** - Parsimonious FBA
   - 최적 성장 유지
   - 총 플럭스 최소화
   - FBA vs pFBA 비교

### Knockout 분석

4. **4_moma_knockout.py** - MOMA Knockout
   - 플럭스 변화 최소화
   - PGI 유전자 knockout 시뮬레이션

5. **5_room_knockout.py** - ROOM Knockout
   - 변화된 반응 수 최소화
   - PGI 유전자 knockout 시뮬레이션

### 생산 분석

6. **6_fseof_analysis.py** - FSEOF 분석
   - 숙신산 생산 범위 탐색
   - 제약된 성장 조건 분석

7. **7_fvseof_analysis.py** - FVSEOF 분석
   - 여러 생산 수준 분석
   - 대사 유연성 평가

8. **8_custom_objective.py** - 사용자 정의 목적 함수
   - 숙신산, 아세트산 생산 최대화
   - ATP 유지 비용 최소화

9. **9_knockout_production_analysis.py** - 종합 Knockout 분석
   - MOMA와 FBA 결합 분석
   - 성장과 생산 trade-off 분석
   - 결과 CSV 저장

### Jupyter 튜토리얼

- **Tutorial_1_FBA_Basics.ipynb** - FBA 기초 튜토리얼
  - 대화형 Jupyter 노트북
  - 시각화 포함
  - 단계별 설명

## 📊 결과 확인

일부 예제(특히 9번)는 `results/` 폴더에 CSV 파일을 저장합니다:

```
results/
├── knockout_analysis_YYYYMMDD_HHMMSS.csv
└── knockout_summary_YYYYMMDD_HHMMSS.csv
```

## ⚠️ 주의사항

- 모든 예제는 **pixi 환경에서 실행**해야 합니다
- 7번과 9번 예제는 실행 시간이 오래 걸릴 수 있습니다
- Jupyter에서는 **Python 3 (ipykernel)** 커널을 선택하세요

## 🔧 문제 해결

### 모듈을 찾을 수 없음 (ModuleNotFoundError)

```bash
# Pixi 환경이 활성화되었는지 확인
pixi shell
```

### Simulator를 찾을 수 없음

```bash
# simulator_tutorials 폴더에서 실행하는지 확인
cd simulator_tutorials
```

### Jupyter kernel 오류

```bash
# Kernel 목록 확인
pixi run jupyter kernelspec list

# Python 3 (ipykernel) 이 있어야 함
```

## 📖 더 자세한 정보

전체 문서는 상위 폴더의 `Simulator_USAGE.md`를 참조하세요.

## 🆘 도움말

문제가 발생하면:
1. `Simulator_USAGE.md`의 설치 가이드 확인
2. Pixi 환경이 활성화되었는지 확인
3. 필요한 모든 파일(Simulator.py, iML1515.xml)이 있는지 확인

