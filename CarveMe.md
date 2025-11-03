# 대장균 대사 모델 구축 가이드

이 문서는 CarveMe를 사용하여 대장균(*E. coli*) 게놈 규모 대사 모델(Genome-scale Metabolic Model, GEM)을 구축하는 전체 과정을 설명합니다.

## 목차
1. [개요](#개요)
2. [대장균 단백질 서열 다운로드](#대장균-단백질-서열-다운로드)
3. [Pixi 환경 구성](#pixi-환경-구성)
4. [CPLEX 솔버 설치](#cplex-솔버-설치)
5. [CarveMe 모델 구축](#carveme-모델-구축)
6. [모델 Gap-filling](#모델-gap-filling)
7. [추가 분석 및 검증](#추가-분석-및-검증)

## 개요

게놈 규모 대사 모델(GEM)은 생물체의 전체 대사 네트워크를 수학적으로 표현한 것으로, 시스템 생물학 연구와 대사공학에 필수적인 도구입니다. 이 가이드에서는 CarveMe 도구를 사용하여 대장균의 게놈 서열로부터 자동으로 GEM을 구축하고, gap-filling을 통해 모델을 완성하는 과정을 다룹니다.

### 주요 도구
- **CarveMe**: 게놈 서열로부터 자동으로 대사 모델을 생성하는 도구
- **COBRApy**: Python 기반 대사 모델 분석 패키지
- **Pixi**: 재현 가능한 환경 관리를 위한 패키지 매니저
- **MEMOTE**: 대사 모델 품질 평가 도구

## 대장균 단백질 서열 다운로드

NCBI에서 *E. coli* K-12 MG1655 균주의 단백질 서열 파일을 다운로드합니다.

```bash
# NCBI FTP 서버에서 대장균 단백질 FASTA 파일 다운로드
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/010/245/GCA_000010245.1_ASM1024v1/GCA_000010245.1_ASM1024v1_protein.faa.gz

# 압축 해제
gunzip GCA_000010245.1_ASM1024v1_protein.faa.gz
```

**참고**: *E. coli* K-12 MG1655는 실험실에서 가장 널리 사용되는 대장균 모델 균주로, 완전히 서열분석된 게놈을 가지고 있습니다.

## Pixi 환경 구성

재현 가능한 연구 환경을 위해 Pixi를 사용하여 필요한 패키지를 설치합니다.

```bash
# Pixi 프로젝트 초기화
pixi init gem 
cd gem
```

### pixi.toml 파일 수정

`pixi.toml` 파일을 열어 `[project]` 섹션에 다음 채널 설정을 추가합니다:

```toml
[project]
name = "gem"
channels = ["conda-forge", "bioconda"]
```

**설명**: 
- `conda-forge`: 일반적인 과학 계산 패키지
- `bioconda`: 생물정보학 전용 패키지

### 필수 패키지 설치

```bash
# Python 3.10 설치
pixi add python=3.10

# COBRApy: 대사 모델 분석을 위한 Python 패키지
pixi add --pypi cobra

# CarveMe: 자동 대사 모델 구축 도구
pixi add --pypi carveme

# MEMOTE: 대사 모델 품질 평가 도구
pixi add --pypi memote

# DIAMOND: 고속 단백질 서열 유사도 검색 도구
pixi add diamond

# Pixi 환경 활성화
pixi shell
```

**각 패키지의 역할**:
- **COBRApy**: Flux Balance Analysis(FBA) 등 대사 모델 시뮬레이션 수행
- **CarveMe**: 단백질 서열로부터 대사 반응을 예측하고 모델 생성
- **MEMOTE**: SBML 표준 준수 여부, 질량/전하 균형 등 모델 품질 검증
- **DIAMOND**: BLAST보다 빠른 단백질 상동성 검색으로 대사 효소 예측

## CPLEX 솔버 설치

CPLEX는 선형 계획법(Linear Programming) 문제를 효율적으로 해결하는 상용 솔버로, 대사 모델의 FBA 계산에 사용됩니다.

### CPLEX 설치 과정

1. **IBM CPLEX 다운로드**: IBM Academic Initiative 또는 공식 웹사이트에서 CPLEX를 다운로드합니다.

2. **setup.py 파일 수정**

CPLEX Python API 설치 디렉토리에서 `setup.py` 파일을 찾아 다음과 같이 수정합니다:

```python
def check_version():
    # Pixi 환경 호환성을 위해 버전 체크 임시 비활성화
    return
```

**설명**: Pixi 환경에서 Python 버전 체크가 실패할 수 있어 이를 우회합니다.

3. **CPLEX Python API 설치**

```bash
# CPLEX Python API가 있는 디렉토리로 이동한 후
pip install .
```

**참고**: CPLEX 대신 무료 오픈소스 솔버인 GLPK를 사용할 수도 있지만, CPLEX가 일반적으로 더 빠르고 안정적입니다.

## CarveMe 모델 구축

CarveMe는 단백질 서열 데이터를 기반으로 자동으로 게놈 규모 대사 모델을 생성합니다.

```bash
carve GCA_000010245.1_ASM1024v1_protein.faa \
  --fbc2 \
  -u gramneg \
  -o model.xml
```

**옵션 설명**:
- `--fbc2`: SBML fbc version 2 형식으로 출력 (Flux Balance Constraints)
- `-u gramneg`: 그람 음성균의 세포막 구조를 고려한 모델 생성
- `-o model.xml`: 출력 파일명 지정

**처리 과정**:
1. DIAMOND를 사용하여 단백질 서열을 BiGG 데이터베이스와 비교
2. 상동성 기반으로 대사 효소를 예측
3. 예측된 효소에 해당하는 대사 반응을 모델에 추가
4. 생합성 경로(biosynthesis) 및 전송 반응 자동 추가

## 모델 Gap-filling

초기 생성된 모델에는 불완전한 대사 경로가 있을 수 있습니다. Gap-filling은 특정 배지 조건에서 생장이 가능하도록 누락된 반응을 채워 넣는 과정입니다.

```bash
gapfill model.xml \
  -m M9 \
  -o new_model.xml
```

**옵션 설명**:
- `-m M9`: M9 최소배지 조건에서 생장 가능하도록 gap-filling 수행
- `-o new_model.xml`: Gap-filling이 완료된 모델 저장

**M9 배지**: 포도당, 무기염류 등 최소한의 영양소만 포함한 합성배지로, 대장균 연구에서 표준적으로 사용됩니다.

## 추가 분석 및 검증

구축된 모델의 품질을 평가하고 추가 gap-filling을 수행합니다.

### 분석 도구 다운로드

```bash
# 대사 모델 분석 스크립트 저장소 클론
git clone https://github.com/jyryu3161/bio_system_design.git
```

### 1. 초기 모델 분석

```bash
# 모델의 기본 통계 및 biomass 반응 확인
python ./bio_system_design/cobrapy_report.py \
  new_model.xml \
  new_model_biomass.xml >> output.txt
```

**분석 내용**:
- 모델에 포함된 대사물질(metabolites), 반응(reactions), 유전자(genes) 수
- Biomass 반응 조성
- 기본 FBA 시뮬레이션 결과

### 2. M9 배지 조건 설정

```bash
# M9 최소배지 조건으로 교환 반응 설정
python ./bio_system_design/set_M9_medium.py \
  new_model_biomass.xml \
  new_model_biomass_M9.xml >> output.txt
```

**기능**: 모델의 교환 반응(exchange reactions)을 M9 배지 조성에 맞게 조정하여 현실적인 배지 조건을 반영합니다.

### 3. 참조 모델 기반 Gap-filling

```bash
# iML1515 참조 모델을 사용한 추가 gap-filling
python ./bio_system_design/run_gapfilling.py \
  new_model_biomass_M9.xml \
  ./bio_system_design/iML1515.xml \
  new_model_biomass_M9_gapfill.xml
```

**설명**:
- **iML1515**: 수작업으로 큐레이션된 고품질 대장균 대사 모델
- 참조 모델의 반응을 활용하여 누락된 경로를 보완
- 최종적으로 생장 가능한 완전한 대사 모델 생성

### 결과 파일

분석 과정에서 생성되는 주요 파일:
- `new_model_biomass.xml`: Biomass 반응이 확인/조정된 모델
- `new_model_biomass_M9.xml`: M9 배지 조건이 설정된 모델
- `new_model_biomass_M9_gapfill.xml`: 최종 gap-filling이 완료된 모델
- `output.txt`: 각 단계별 분석 결과 로그

## 추가 권장 사항

### 모델 품질 평가

```bash
# MEMOTE를 사용한 포괄적인 품질 평가
memote report snapshot new_model_biomass_M9_gapfill.xml --filename report.html
```

### 모델 활용 예시

완성된 모델로 수행할 수 있는 분석:
- **FBA (Flux Balance Analysis)**: 특정 조건에서 대사 흐름 예측
- **유전자 필수성 분석**: 특정 유전자 결실이 생장에 미치는 영향 예측
- **대사 엔지니어링 설계**: 목표 물질 생산을 위한 최적 경로 탐색
- **약물 표적 발굴**: 필수 대사 경로 식별

### 참고 문헌

- **CarveMe**: Machado et al. (2018), Nucleic Acids Research
- **COBRApy**: Ebrahim et al. (2013), BMC Systems Biology
- **iML1515**: Monk et al. (2017), Nature Biotechnology

---

**문제 해결 팁**:
- Gap-filling이 실패하는 경우: 배지 조성을 확인하거나 더 관대한 조건(`-m LB`)을 사용
- CPLEX 설치 문제: 오픈소스 솔버(GLPK)로 대체 가능
- 모델이 생장하지 않는 경우: `memote` 보고서에서 질량 불균형이나 막힌 반응 확인
