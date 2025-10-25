# Metabolic Model Simulator

A Python-based constraint-based metabolic modeling simulator implementing Flux Balance Analysis (FBA), Minimization of Metabolic Adjustment (MOMA), and Regulatory On/Off Minimization (ROOM) methods.

## Overview

This simulator provides comprehensive tools for analyzing genome-scale metabolic models using various constraint-based methods:

- **FBA (Flux Balance Analysis)**: Predicts metabolic flux distributions by optimizing an objective function
- **pFBA (Parsimonious FBA)**: Minimizes total flux while maintaining optimal growth
- **MOMA (Minimization of Metabolic Adjustment)**: Predicts metabolic behavior after genetic perturbations by minimizing flux changes
- **ROOM (Regulatory On/Off Minimization)**: Predicts knockout phenotypes by minimizing the number of significantly changed reactions

## Features

- ✅ Load models from SBML files or COBRApy model objects
- ✅ Perform standard and parsimonious FBA
- ✅ Simulate gene knockouts using MOMA or ROOM
- ✅ Flexible flux constraints and custom objectives
- ✅ Comprehensive documentation and examples

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Install COBRApy

COBRApy is a Python package for constraint-based metabolic modeling.

```bash
pip install cobra
```

Or install with conda:

```bash
conda install -c conda-forge cobra
```

### Step 2: Install Gurobi

#### Option A: Academic/Free License

1. **Register for a free academic license** at [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)

2. **Install Gurobi**:

```bash
pip install gurobipy
```

Or with conda:

```bash
conda install -c gurobi gurobi
```

3. **Activate your license**:

After obtaining your license key from Gurobi:

```bash
grbgetkey YOUR-LICENSE-KEY
```

#### Option B: Free Limited License

Gurobi offers a free limited license (for models with up to 2,000 variables):

```bash
pip install gurobipy
```

No license activation required for the limited version.

#### Option C: Alternative Solvers (Optional)

If you don't want to use Gurobi, you can modify the code to use other solvers supported by COBRApy:

- **GLPK** (free, open-source)
- **CPLEX** (commercial, free for academics)
- **SCIP** (free, open-source)

### Step 3: Verify Installation

```python
import cobra
import gurobipy
print(f"COBRApy version: {cobra.__version__}")
print(f"Gurobi version: {gurobipy.gurobi.version()}")
```

Expected output:
```
COBRApy version: 0.29.0
Gurobi version: (11, 0, 0)
```

## Quick Start

### 1. Basic FBA Analysis

```python
from simulator_improved import Simulator

# Initialize simulator
sim = Simulator()

# Load a metabolic model (SBML format)
sim.read_model('path/to/model.xml')

# Run FBA
status, objective_value, flux_distribution = sim.run_FBA()

if status == 2:  # Optimal solution found
    print(f"Optimal growth rate: {objective_value:.4f}")
    print(f"Number of active reactions: {sum(1 for v in flux_distribution.values() if abs(v) > 1e-6)}")
```

### 2. Using COBRApy Test Models

```python
from simulator_improved import Simulator
from cobra.test import create_test_model

# Load E. coli core model
sim = Simulator()
model = create_test_model("textbook")
sim.load_cobra_model(model)

# Run FBA
status, growth_rate, fluxes = sim.run_FBA()
print(f"Growth rate: {growth_rate:.4f} hr⁻¹")
```

### 3. Parsimonious FBA (pFBA)

```python
# Minimize total flux while maintaining optimal growth
status, total_flux, fluxes = sim.run_FBA(internal_flux_minimization=True)

if status == 2:
    print(f"Total flux sum: {total_flux:.4f}")
```

### 4. Gene Knockout Simulation with MOMA

```python
from simulator_improved import Simulator
from cobra.test import create_test_model

# Load model and get wild-type flux distribution
sim = Simulator()
model = create_test_model("textbook")
sim.load_cobra_model(model)

# Get wild-type (reference) flux distribution
_, wt_growth, wt_fluxes = sim.run_FBA()
print(f"Wild-type growth rate: {wt_growth:.4f}")

# Simulate PGI gene knockout
knockout_constraints = {'PGI': (0, 0)}  # Block PGI reaction

# Run MOMA to predict knockout phenotype
status, distance, moma_fluxes = sim.run_MOMA(
    wild_flux=wt_fluxes,
    flux_constraints=knockout_constraints
)

if status == 2:
    knockout_growth = moma_fluxes[sim.objective]
    print(f"MOMA predicted growth rate: {knockout_growth:.4f}")
    print(f"Growth reduction: {(1 - knockout_growth/wt_growth)*100:.1f}%")
    print(f"Metabolic distance: {distance:.4f}")
```

### 5. Gene Knockout Simulation with ROOM

```python
# Run ROOM to predict knockout phenotype
status, n_changed, room_fluxes = sim.run_ROOM(
    wild_flux=wt_fluxes,
    flux_constraints=knockout_constraints,
    delta=0.03,  # 3% relative tolerance
    epsilon=0.001  # Absolute tolerance for near-zero fluxes
)

if status == 2:
    knockout_growth = room_fluxes[sim.objective]
    print(f"ROOM predicted growth rate: {knockout_growth:.4f}")
    print(f"Number of significantly changed reactions: {int(n_changed)}")
```

### 6. Custom Objective Function

```python
# Maximize production of a specific metabolite
status, production_rate, fluxes = sim.run_FBA(
    new_objective='EX_succ_e',  # Succinate exchange reaction
    mode='max'
)

print(f"Maximum succinate production: {production_rate:.4f}")
```

### 7. Flux Variability Analysis

```python
# Find flux range for each reaction while maintaining optimal growth
_, optimal_growth, _ = sim.run_FBA()

flux_ranges = {}
for reaction in sim.model_reactions:
    # Fix growth at optimal value
    growth_constraint = {sim.objective: (optimal_growth * 0.99, optimal_growth)}
    
    # Minimize reaction flux
    _, min_flux, min_fluxes = sim.run_FBA(
        new_objective=reaction,
        flux_constraints=growth_constraint,
        mode='min'
    )
    
    # Maximize reaction flux
    _, max_flux, max_fluxes = sim.run_FBA(
        new_objective=reaction,
        flux_constraints=growth_constraint,
        mode='max'
    )
    
    flux_ranges[reaction] = (min_flux, max_flux)

# Find essential reactions (those that must carry flux)
essential_reactions = [r for r, (min_f, max_f) in flux_ranges.items() 
                       if min_f > 1e-6 or max_f < -1e-6]
print(f"Number of essential reactions: {len(essential_reactions)}")
```

### 8. Multiple Gene Knockout Analysis

```python
import pandas as pd

# Test multiple gene knockouts
genes_to_test = ['PGI', 'PFK', 'FBA', 'TPI', 'GAPD']
results = []

for gene in genes_to_test:
    # Create knockout constraint
    knockout = {gene: (0, 0)}
    
    # Predict with MOMA
    status, dist, moma_flux = sim.run_MOMA(wt_fluxes, knockout)
    moma_growth = moma_flux[sim.objective] if status == 2 else 0
    
    # Predict with ROOM
    status, n_changed, room_flux = sim.run_ROOM(wt_fluxes, knockout)
    room_growth = room_flux[sim.objective] if status == 2 else 0
    
    results.append({
        'Gene': gene,
        'WT Growth': wt_growth,
        'MOMA Growth': moma_growth,
        'ROOM Growth': room_growth,
        'MOMA Reduction (%)': (1 - moma_growth/wt_growth) * 100,
        'ROOM Reduction (%)': (1 - room_growth/wt_growth) * 100,
        'Reactions Changed': n_changed if status == 2 else 'N/A'
    })

# Display results
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 9. Comparing Wild-type and Knockout Flux Distributions

```python
import numpy as np
import matplotlib.pyplot as plt

# Get knockout predictions
knockout = {'PGI': (0, 0)}
_, _, moma_flux = sim.run_MOMA(wt_fluxes, knockout)

# Compare flux distributions
reactions = list(wt_fluxes.keys())
wt_values = [wt_fluxes[r] for r in reactions]
ko_values = [moma_flux[r] for r in reactions]

# Calculate flux differences
differences = [abs(wt_values[i] - ko_values[i]) for i in range(len(reactions))]

# Find top 10 most changed reactions
top_changed = sorted(zip(reactions, differences), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 most changed reactions:")
for reaction, diff in top_changed:
    wt_val = wt_fluxes[reaction]
    ko_val = moma_flux[reaction]
    print(f"{reaction:20s} WT: {wt_val:8.4f}  KO: {ko_val:8.4f}  Diff: {diff:8.4f}")
```

## Advanced Usage

### Loading Models from Different Sources

```python
from simulator_improved import Simulator
import cobra

# From SBML file
sim1 = Simulator()
sim1.read_model('path/to/model.xml')

# From COBRApy model object
sim2 = Simulator()
cobra_model = cobra.io.load_model("textbook")
sim2.load_cobra_model(cobra_model)

# From BiGG Models database
sim3 = Simulator()
model = cobra.io.load_model("iJO1366")  # E. coli iJO1366 model
sim3.load_cobra_model(model)
```

### Custom Flux Constraints

```python
# Set glucose uptake rate
constraints = {
    'EX_glc__D_e': (-10, -10),  # Fix glucose uptake at 10 mmol/gDW/hr
    'EX_o2_e': (-20, 0),        # Limit oxygen uptake
    'ATPM': (8.39, 8.39)        # Fix ATP maintenance
}

status, growth, fluxes = sim.run_FBA(flux_constraints=constraints)
```

### Working with Infinite Bounds

```python
# Use actual infinite bounds (not replaced with ±1000)
status, growth, fluxes = sim.run_FBA(inf_flag=True)
```

## Method Comparison

| Method | Optimization Type | Use Case | Computation Time |
|--------|------------------|----------|------------------|
| **FBA** | Linear Programming (LP) | Predict optimal flux distribution | Fast (seconds) |
| **pFBA** | LP (two-stage) | More realistic flux distribution | Fast (seconds) |
| **MOMA** | Quadratic Programming (QP) | Predict immediate knockout response | Medium (seconds) |
| **ROOM** | Mixed-Integer LP (MILP) | Predict regulatory knockout response | Slow (minutes) |

### When to Use Each Method

- **FBA**: Predicting maximum growth rate or production under different conditions
- **pFBA**: Getting more realistic flux distributions without futile cycles
- **MOMA**: Simulating immediate response to gene knockout (before regulation)
- **ROOM**: Simulating long-term adapted response to gene knockout (after regulation)

## API Reference

### Simulator Class

#### `__init__()`
Initialize a new Simulator instance.

#### `read_model(filename)`
Load a metabolic model from an SBML file.

**Parameters:**
- `filename` (str): Path to SBML model file

**Returns:** Model components tuple

#### `load_cobra_model(cobra_model)`
Load a COBRApy model object.

**Parameters:**
- `cobra_model` (cobra.Model): COBRApy model object

**Returns:** Model components tuple

#### `run_FBA(new_objective='', flux_constraints={}, inf_flag=False, internal_flux_minimization=False, mode='max')`
Perform Flux Balance Analysis.

**Parameters:**
- `new_objective` (str): Reaction ID for objective (default: model's objective)
- `flux_constraints` (dict): Additional flux constraints {reaction: (lb, ub)}
- `inf_flag` (bool): Keep infinite bounds if True
- `internal_flux_minimization` (bool): Perform pFBA if True
- `mode` (str): 'max' or 'min'

**Returns:** (status, objective_value, flux_distribution)

#### `run_MOMA(wild_flux={}, flux_constraints={}, inf_flag=False)`
Perform Minimization of Metabolic Adjustment.

**Parameters:**
- `wild_flux` (dict): Reference flux distribution
- `flux_constraints` (dict): Additional flux constraints {reaction: (lb, ub)}
- `inf_flag` (bool): Keep infinite bounds if True

**Returns:** (status, distance, flux_distribution)

#### `run_ROOM(wild_flux={}, flux_constraints={}, delta=0.03, epsilon=0.001, inf_flag=False)`
Perform Regulatory On/Off Minimization.

**Parameters:**
- `wild_flux` (dict): Reference flux distribution
- `flux_constraints` (dict): Additional flux constraints {reaction: (lb, ub)}
- `delta` (float): Relative tolerance for flux changes (default: 0.03)
- `epsilon` (float): Absolute tolerance for near-zero fluxes (default: 0.001)
- `inf_flag` (bool): Keep infinite bounds if True

**Returns:** (status, n_changed_reactions, flux_distribution)

## Troubleshooting

### Gurobi License Issues

**Problem:** `GurobiError: No Gurobi license found`

**Solution:**
1. Verify license installation: `gurobi_cl --license`
2. Check license file location: `~/gurobi.lic` (Linux/Mac) or `C:\gurobi\gurobi.lic` (Windows)
3. Request a new academic license if expired

### Model Loading Errors

**Problem:** `FileNotFoundError` or `CobraSBMLError`

**Solution:**
1. Verify file path is correct
2. Ensure SBML file is valid (test with COBRApy directly)
3. Check file permissions

### Infeasible Solutions

**Problem:** Solver returns status != 2 (infeasible or unbounded)

**Solution:**
1. Check flux constraints for conflicts
2. Verify model has no gaps (dead-end metabolites)
3. Inspect exchange reactions and medium composition
4. Use `cobra.io.load_model()` first to validate model

### Memory Issues with Large Models

**Problem:** Out of memory errors with genome-scale models

**Solution:**
1. Use pFBA instead of regular FBA when possible
2. Reduce number of reactions in flux variability analysis
3. Increase system RAM or use computing cluster

## Performance Tips

1. **Use inf_flag=False** for faster optimization (replaces infinite bounds with ±1000)
2. **Avoid unnecessary ROOM analyses** - it's the slowest method due to MILP
3. **Reuse wild-type flux distributions** instead of recalculating for each knockout
4. **Pre-filter reactions** before flux variability analysis to reduce computation time

## Citation

If you use this simulator in your research, please cite:

### FBA
- Orth, J. D., Thiele, I., & Palsson, B. Ø. (2010). What is flux balance analysis? *Nature Biotechnology*, 28(3), 245-248.

### MOMA
- Segrè, D., Vitkup, D., & Church, G. M. (2002). Analysis of optimality in natural and perturbed metabolic networks. *Proceedings of the National Academy of Sciences*, 99(23), 15112-15117.

### ROOM
- Shlomi, T., Berkman, O., & Ruppin, E. (2005). Regulatory on/off minimization of metabolic flux changes after genetic perturbations. *Proceedings of the National Academy of Sciences*, 102(21), 7695-7700.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- COBRApy development team
- Gurobi Optimization
- Systems Biology research community
