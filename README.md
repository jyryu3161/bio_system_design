# Design of Biological Systems

## Simulator

📖 **[Complete Usage Guide & Examples →](USAGE.md)**

### Quick Start

#### Installation

1. **Install COBRApy:**
```bash
pip install cobra
```

2. **Install Gurobi:**
```bash
pip install gurobipy
```

For academic use, get a free license at [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)

3. **Verify installation:**
```python
import cobra
import gurobipy
print(f"COBRApy version: {cobra.__version__}")
print(f"Gurobi version: {gurobipy.gurobi.version()}")
```

#### Basic Example

```python
from simulator import Simulator
from cobra.test import create_test_model

# Load model
sim = Simulator()
model = create_test_model("textbook")
sim.load_cobra_model(model)

# Run FBA
status, growth_rate, fluxes = sim.run_FBA()
print(f"Growth rate: {growth_rate:.4f} hr⁻¹")

# Simulate gene knockout with MOMA
_, _, wt_fluxes = sim.run_FBA()
knockout_constraints = {'PGI': (0, 0)}
status, distance, ko_fluxes = sim.run_MOMA(wt_fluxes, knockout_constraints)
print(f"Knockout growth: {ko_fluxes[sim.objective]:.4f} hr⁻¹")
```
