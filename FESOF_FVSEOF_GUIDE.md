# FSEOF and FVSEOF Implementation Guide

## Overview

이 구현에는 다음 메서드들이 포함되어 있습니다:
- **FBA**: Flux Balance Analysis
- **Linear MOMA**: Minimization of Metabolic Adjustment (Linear version)
- **ROOM**: Regulatory On/Off Minimization
- **FVA**: Flux Variability Analysis
- **FSEOF**: Flux Scanning based on Enforced Objective Flux
- **FVSEOF**: Flux Variability Scanning based on Enforced Objective Flux

## Method Descriptions

### 1. FSEOF (Flux Scanning based on Enforced Objective Flux)

**목적**: 특정 목적함수(예: biomass) 수준을 유지하면서 타겟 반응의 가능한 flux 범위를 찾습니다.

**사용 사례**:
- Production envelope 계산
- 특정 성장률에서의 대사산물 생산 범위 파악
- 대사 bottleneck 식별

**예제**:
```python
import cobra
from simulator_linear_moma import Simulator

# Load model
model = cobra.io.load_model("textbook")
sim = Simulator()
sim.load_cobra_model(model)

# Run FSEOF on acetate export at 90% maximum growth
status, min_flux, max_flux, flux_range = sim.run_FSEOF(
    target_reaction='EX_ac_e',
    objective_fraction=0.9  # 90% of max growth
)

print(f"At 90% growth, acetate flux range: [{min_flux:.4f}, {max_flux:.4f}]")
print(f"Min case biomass: {flux_range['min'][sim.objective]:.4f}")
print(f"Max case biomass: {flux_range['max'][sim.objective]:.4f}")
```

### 2. FVSEOF (Flux Variability Scanning based on Enforced Objective Flux)

**목적**: 타겟 반응의 여러 flux 값에서 전체 네트워크의 flux variability를 분석합니다.

**사용 사례**:
- 생산량에 따른 대사 유연성 변화 파악
- 최적 생산 조건 탐색
- Flux coupling 패턴 분석
- Bioprocess 설계를 위한 operating point 식별

**예제**:
```python
# Run FVSEOF with 10 sampling points
status, results = sim.run_FVSEOF(
    target_reaction='EX_succ_e',
    objective_fraction=0.9,
    num_points=10
)

# Analyze results
for i, target_flux in enumerate(results['target_fluxes']):
    obj_flux = results['objective_fluxes'][i]
    fva_data = results['fva_results'][i]['fva_data']
    
    print(f"\nPoint {i+1}:")
    print(f"  Target flux: {target_flux:.4f}")
    print(f"  Objective range: [{obj_flux['minimum']:.4f}, {obj_flux['maximum']:.4f}]")
    
    # Calculate metabolic flexibility
    ranges = [data['range'] for data in fva_data.values()]
    print(f"  Average flux range: {sum(ranges)/len(ranges):.4f}")
```

### 3. FVA (Flux Variability Analysis)

**목적**: 모든 반응의 최소/최대 가능 flux를 계산합니다.

**예제**:
```python
# Run FVA at 100% optimum
fva_result = sim.run_FVA(
    fraction_of_optimum=1.0,
    reactions_to_analyze=None  # None = all reactions
)

# Check results
for rxn, data in fva_result['fva_data'].items():
    if data['range'] > 0.1:  # Only show reactions with significant variability
        print(f"{rxn}: [{data['minimum']:.4f}, {data['maximum']:.4f}]")
```

## Complete Usage Example

```python
#!/usr/bin/env python3
"""
Complete example demonstrating FSEOF and FVSEOF analysis
for metabolic engineering applications.
"""

import cobra
from simulator_linear_moma import Simulator
import matplotlib.pyplot as plt
import numpy as np

# Load E. coli model
model = cobra.io.load_model("iJO1366")
sim = Simulator()
sim.load_cobra_model(model)

print(f"Model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")
print(f"Objective: {sim.objective}")

# ====================
# 1. Find maximum growth rate
# ====================
status, max_growth, wt_flux = sim.run_FBA()
print(f"\nMaximum growth rate: {max_growth:.4f} /h")

# ====================
# 2. FSEOF Analysis - Production Envelope
# ====================
target_product = 'EX_succ_e'  # Succinate export
growth_fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

print(f"\n{'Growth %':<10} {'Min Flux':<12} {'Max Flux':<12} {'Range':<12}")
print("-" * 50)

envelope_data = []
for fraction in growth_fractions:
    status, min_f, max_f, _ = sim.run_FSEOF(
        target_reaction=target_product,
        objective_fraction=fraction
    )
    
    if status == 'optimal':
        envelope_data.append({
            'fraction': fraction,
            'growth': fraction * max_growth,
            'min': min_f,
            'max': max_f,
            'range': max_f - min_f
        })
        print(f"{fraction*100:<10.1f} {min_f:<12.4f} {max_f:<12.4f} {max_f-min_f:<12.4f}")

# ====================
# 3. FVSEOF Analysis - Metabolic Flexibility
# ====================
print("\nRunning FVSEOF analysis...")
status, fvseof_results = sim.run_FVSEOF(
    target_reaction=target_product,
    objective_fraction=0.9,
    num_points=8
)

if status == 'optimal':
    print(f"✓ Analyzed {len(fvseof_results['target_fluxes'])} points")
    
    # Calculate flexibility at each point
    flexibility_data = []
    for i, fva_result in enumerate(fvseof_results['fva_results']):
        ranges = [data['range'] for data in fva_result['fva_data'].values()]
        avg_range = np.mean(ranges)
        flexibility_data.append(avg_range)
        
        print(f"  Point {i+1}: Target={fvseof_results['target_fluxes'][i]:.4f}, "
              f"Flexibility={avg_range:.4f}")

# ====================
# 4. Visualization
# ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Production envelope
ax1 = axes[0, 0]
growths = [d['growth'] for d in envelope_data]
mins = [d['min'] for d in envelope_data]
maxs = [d['max'] for d in envelope_data]

ax1.fill_between(growths, mins, maxs, alpha=0.3, color='blue', label='Feasible region')
ax1.plot(growths, mins, 'b-', marker='o', label='Minimum', linewidth=2)
ax1.plot(growths, maxs, 'r-', marker='s', label='Maximum', linewidth=2)
ax1.set_xlabel('Growth Rate (/h)', fontsize=12)
ax1.set_ylabel('Succinate Production (mmol/gDW/h)', fontsize=12)
ax1.set_title('Production Envelope (FSEOF)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Trade-off curve
ax2 = axes[0, 1]
target_fluxes = fvseof_results['target_fluxes']
obj_mins = [obj['minimum'] for obj in fvseof_results['objective_fluxes']]
obj_maxs = [obj['maximum'] for obj in fvseof_results['objective_fluxes']]

ax2.fill_between(target_fluxes, obj_mins, obj_maxs, alpha=0.3, color='green')
ax2.plot(target_fluxes, obj_mins, 'g-', marker='o', label='Min growth', linewidth=2)
ax2.plot(target_fluxes, obj_maxs, 'g-', marker='s', label='Max growth', linewidth=2)
ax2.set_xlabel('Succinate Production (mmol/gDW/h)', fontsize=12)
ax2.set_ylabel('Growth Rate (/h)', fontsize=12)
ax2.set_title('Growth-Production Trade-off (FVSEOF)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Metabolic flexibility
ax3 = axes[1, 0]
ax3.plot(target_fluxes, flexibility_data, 'purple', marker='D', linewidth=2, markersize=8)
ax3.set_xlabel('Succinate Production (mmol/gDW/h)', fontsize=12)
ax3.set_ylabel('Average Flux Variability', fontsize=12)
ax3.set_title('Metabolic Flexibility (FVSEOF)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Range vs Growth
ax4 = axes[1, 1]
fractions = [d['fraction'] for d in envelope_data]
ranges = [d['range'] for d in envelope_data]

ax4.plot(fractions, ranges, 'orange', marker='*', linewidth=2, markersize=10)
ax4.set_xlabel('Fraction of Maximum Growth', fontsize=12)
ax4.set_ylabel('Production Range (mmol/gDW/h)', fontsize=12)
ax4.set_title('Production Flexibility vs Growth', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fseof_fvseof_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'fseof_fvseof_comprehensive_analysis.png'")

# ====================
# 5. Identify optimal operating point
# ====================
print("\n" + "="*60)
print("OPTIMAL OPERATING POINT ANALYSIS")
print("="*60)

# Find point with highest production range and reasonable growth
optimal_idx = None
max_range = 0

for i, d in enumerate(envelope_data):
    if d['fraction'] >= 0.5 and d['range'] > max_range:
        max_range = d['range']
        optimal_idx = i

if optimal_idx is not None:
    opt = envelope_data[optimal_idx]
    print(f"\nRecommended operating point:")
    print(f"  Growth rate: {opt['growth']:.4f} /h ({opt['fraction']*100:.0f}% of maximum)")
    print(f"  Production range: {opt['min']:.4f} to {opt['max']:.4f} mmol/gDW/h")
    print(f"  Flexibility: {opt['range']:.4f} mmol/gDW/h")

print("\n" + "="*60)
```

## Running the Tests

```bash
# Run all tests with textbook model (fast)
python simulator_linear_moma.py

# Run only FSEOF/FVSEOF tests
python simulator_linear_moma.py fseof textbook

# Run comparison tests with larger model
python simulator_linear_moma.py comparison iJO1366

# Run all tests with E. coli core model
python simulator_linear_moma.py all e_coli_core
```

## Applications

### 1. Metabolic Engineering
- 타겟 대사산물의 최대 생산 가능량 예측
- 성장률과 생산량 간의 trade-off 분석
- 최적 생산 조건 탐색

### 2. Strain Design
- Gene knockout 후 생산량 변화 예측
- 대사 경로 재설계 효과 평가
- Flux redistribution 분석

### 3. Bioprocess Optimization
- 최적 operating point 식별
- Fed-batch 전략 설계
- 대사 병목 현상 파악

### 4. Systems Biology Research
- 대사 네트워크 유연성 분석
- Flux coupling 패턴 연구
- Metabolic regulation 메커니즘 이해

## Key Parameters

### FSEOF
- `target_reaction`: 분석할 반응 ID
- `objective_fraction`: 유지할 목적함수의 비율 (0-1)
- `flux_constraints`: 추가 flux 제약조건

### FVSEOF
- `target_reaction`: 스캔할 반응 ID
- `objective_fraction`: 유지할 목적함수의 비율
- `num_points`: 샘플링할 포인트 개수
- `flux_constraints`: 추가 flux 제약조건

### FVA
- `fraction_of_optimum`: 유지할 최적값의 비율
- `reactions_to_analyze`: 분석할 반응 리스트 (None=모두)

## Performance Tips

1. **FVA 속도 향상**: `reactions_to_analyze`로 관심 반응만 선택
2. **FVSEOF 포인트 수**: 상세 분석은 20+, 빠른 스캔은 5-10 포인트
3. **큰 모델**: iJO1366 같은 대형 모델은 시간이 오래 걸릴 수 있음
4. **병렬화**: 필요시 각 포인트를 병렬로 처리 가능

## References

1. Mahadevan, R., & Schilling, C. H. (2003). The effects of alternate optimal solutions in constraint-based genome-scale metabolic models. *Metabolic Engineering*, 5(4), 264-276.

2. Choi, H. S., Lee, S. Y., Kim, T. Y., & Woo, H. M. (2010). In silico identification of gene amplification targets for improvement of lycopene production. *Applied and Environmental Microbiology*, 76(10), 3097-3105.

3. Burgard, A. P., Pharkya, P., & Maranas, C. D. (2003). Optknock: a bilevel programming framework for identifying gene knockout strategies for microbial strain optimization. *Biotechnology and Bioengineering*, 84(6), 647-657.
