import cobra.io as io
from gurobipy import *


class Simulator(object):
    """
    Constraint-based metabolic model simulator implementing FBA, MOMA, and ROOM methods.
    
    This class provides methods for:
    - Flux Balance Analysis (FBA)
    - Minimization of Metabolic Adjustment (MOMA)
    - Regulatory On/Off Minimization (ROOM)
    """
    
    def __init__(self):
        """
        Constructor for Simulator
        
        Initializes all model components to None. These will be populated
        when a model is loaded using read_model() or load_cobra_model().
        """
        self.cobra_model = None
        self.model_metabolites = None
        self.model_reactions = None
        self.model_genes = None  # Added to avoid AttributeError
        self.Smatrix = None
        self.lower_boundary_constraints = None
        self.upper_boundary_constraints = None
        self.objective = None

    def run_MOMA(self, wild_flux={}, flux_constraints={}, inf_flag=False):
        """
        Minimization of Metabolic Adjustment (MOMA) analysis.
        
        MOMA finds a flux distribution that minimizes the Euclidean distance
        to a reference (wild-type) flux distribution while satisfying stoichiometric
        and flux bound constraints.
        
        Parameters:
        -----------
        wild_flux : dict
            Reference flux distribution (typically wild-type). Keys are reaction IDs,
            values are flux values.
        flux_constraints : dict
            Additional flux constraints. Keys are reaction IDs, values are tuples
            of (lower_bound, upper_bound).
        inf_flag : bool, optional (default=False)
            If False, replaces infinite bounds with ±1000.
            
        Returns:
        --------
        tuple: (status, objective_value, flux_distribution)
            - status: Gurobi optimization status (2 = optimal)
            - objective_value: Minimized distance metric
            - flux_distribution: Dict of reaction IDs to flux values
            
        Notes:
        ------
        The objective minimizes: sum((v - w)^2) where v is mutant flux and w is wild-type flux.
        This is linearized using auxiliary variables fplus and fminus.
        """
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions
        Smatrix = self.Smatrix
        lower_boundary_constraints = self.lower_boundary_constraints.copy()
        upper_boundary_constraints = self.upper_boundary_constraints.copy()

        # Replace infinite bounds with finite values if inf_flag is False
        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        # Prepare Gurobi data structures
        pairs, coffvalue = multidict(Smatrix)
        pairs = tuplelist(pairs)

        # Create Gurobi model
        m = Model('MOMA')
        m.setParam('OutputFlag', 0)
        m.reset()

        # Create variables
        v = {}  # Flux variables
        fplus = {}  # Positive deviation from wild-type
        fminus = {}  # Negative deviation from wild-type

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], 
                                            name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            # Deviation variables (non-negative)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f"fplus_{each_reaction}")
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f"fminus_{each_reaction}")

        m.update()

        # Add constraints relating flux to deviations
        for each_reaction in model_reactions:
            # v = fplus - fminus (flux decomposition)
            m.addConstr(v[each_reaction] == (fplus[each_reaction] - fminus[each_reaction]),
                       name=f"flux_decomp_{each_reaction}")
            
            # fplus >= v - wild_flux (captures positive deviation)
            if each_reaction in wild_flux:
                m.addConstr(fplus[each_reaction] >= v[each_reaction] - wild_flux[each_reaction],
                           name=f"fplus_lb_{each_reaction}")
                # fminus >= wild_flux - v (captures negative deviation)
                m.addConstr(fminus[each_reaction] >= wild_flux[each_reaction] - v[each_reaction],
                           name=f"fminus_lb_{each_reaction}")

        m.update()

        # Add steady-state mass balance constraints (Sv = 0)
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                v[reaction] * coffvalue[metabolite, reaction] 
                for metabolite, reaction in pairs.select(each_metabolite, '*')) == 0,
                name=f"mass_balance_{each_metabolite}")

        m.update()

        # Set objective: minimize sum of absolute deviations
        # This approximates Euclidean distance minimization
        target_reactions = wild_flux.keys()
        m.setObjective(quicksum(
            (fplus[each_reaction] + fminus[each_reaction]) 
            for each_reaction in target_reactions), GRB.MINIMIZE)

        # Solve the optimization problem
        m.optimize()

        # Extract results if optimization was successful
        if m.status == 2:  # GRB.OPTIMAL
            flux_distribution = {}
            for reaction in model_reactions:
                flux_distribution[reaction] = float(v[reaction].x)
                # Round near-zero values to exactly zero
                if abs(float(v[reaction].x)) <= 1e-6:
                    flux_distribution[reaction] = 0.0

            return m.status, m.ObjVal, flux_distribution
        else:
            return m.status, False, False

    def run_ROOM(self, wild_flux={}, flux_constraints={}, delta=0.03, epsilon=0.001, inf_flag=False):
        """
        Regulatory On/Off Minimization (ROOM) analysis.
        
        ROOM finds a flux distribution that minimizes the number of significant flux changes
        compared to a reference (wild-type) flux distribution. It uses binary variables to
        count reactions that deviate significantly from the reference state.
        
        Parameters:
        -----------
        wild_flux : dict
            Reference flux distribution (typically wild-type). Keys are reaction IDs,
            values are flux values.
        flux_constraints : dict
            Additional flux constraints. Keys are reaction IDs, values are tuples
            of (lower_bound, upper_bound).
        delta : float, optional (default=0.03)
            Relative tolerance for flux changes. A reaction is considered "changed"
            if |v - w| > delta * |w| where v is mutant flux and w is wild-type flux.
        epsilon : float, optional (default=0.001)
            Absolute tolerance for near-zero fluxes. Used to avoid division by zero
            and to define inactive reactions.
        inf_flag : bool, optional (default=False)
            If False, replaces infinite bounds with ±1000.
            
        Returns:
        --------
        tuple: (status, objective_value, flux_distribution)
            - status: Gurobi optimization status (2 = optimal)
            - objective_value: Number of significantly changed reactions
            - flux_distribution: Dict of reaction IDs to flux values
            
        Notes:
        ------
        ROOM is particularly useful for predicting gene knockout phenotypes, as it
        assumes the cell minimizes regulatory changes rather than metabolic adjustment.
        """
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions
        Smatrix = self.Smatrix
        lower_boundary_constraints = self.lower_boundary_constraints.copy()
        upper_boundary_constraints = self.upper_boundary_constraints.copy()

        # Replace infinite bounds with finite values if inf_flag is False
        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        # Prepare Gurobi data structures
        pairs, coffvalue = multidict(Smatrix)
        pairs = tuplelist(pairs)

        # Create Gurobi model
        m = Model('ROOM')
        m.setParam('OutputFlag', 0)
        m.reset()

        # Create variables
        v = {}  # Flux variables
        y = {}  # Binary variables: 1 if reaction flux significantly changed, 0 otherwise

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], 
                                            name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            
            # Binary indicator variable for significant flux change
            y[each_reaction] = m.addVar(vtype=GRB.BINARY, name=f"y_{each_reaction}")

        m.update()

        # Add steady-state mass balance constraints (Sv = 0)
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                v[reaction] * coffvalue[metabolite, reaction] 
                for metabolite, reaction in pairs.select(each_metabolite, '*')) == 0,
                name=f"mass_balance_{each_metabolite}")

        m.update()

        # Add ROOM-specific constraints
        # For each reaction with reference flux, constrain deviation based on binary variable
        for each_reaction in model_reactions:
            if each_reaction not in wild_flux:
                continue
                
            w = wild_flux[each_reaction]  # Reference (wild-type) flux
            
            # Calculate upper and lower bounds for allowable flux range
            if abs(w) < epsilon:
                # If wild-type flux is near zero, use absolute tolerance
                w_upper = epsilon
                w_lower = -epsilon
            else:
                # Use relative tolerance based on wild-type flux magnitude
                w_upper = w + delta * abs(w)
                w_lower = w - delta * abs(w)
            
            # Big-M formulation to link binary variable y to flux deviation
            # If y = 0 (no significant change), then w_lower <= v <= w_upper
            # If y = 1 (significant change allowed), then lb <= v <= ub (no additional constraint)
            
            # Get actual bounds for the reaction
            if each_reaction in flux_constraints:
                lb = flux_constraints[each_reaction][0]
                ub = flux_constraints[each_reaction][1]
            else:
                lb = lower_boundary_constraints[each_reaction]
                ub = upper_boundary_constraints[each_reaction]
            
            # Big-M values (should be larger than any feasible flux)
            M_upper = ub - w_upper if ub < 1000 else 2000
            M_lower = w_lower - lb if lb > -1000 else 2000
            
            # v <= w_upper + M_upper * y
            # When y=0: v <= w_upper (enforces upper bound of allowable range)
            # When y=1: v <= w_upper + M_upper (effectively no constraint)
            m.addConstr(v[each_reaction] <= w_upper + M_upper * y[each_reaction],
                       name=f"room_upper_{each_reaction}")
            
            # v >= w_lower - M_lower * y
            # When y=0: v >= w_lower (enforces lower bound of allowable range)
            # When y=1: v >= w_lower - M_lower (effectively no constraint)
            m.addConstr(v[each_reaction] >= w_lower - M_lower * y[each_reaction],
                       name=f"room_lower_{each_reaction}")

        m.update()

        # Set objective: minimize number of significantly changed reactions
        m.setObjective(quicksum(y[each_reaction] for each_reaction in wild_flux.keys()), 
                      GRB.MINIMIZE)

        # Solve the optimization problem
        m.optimize()

        # Extract results if optimization was successful
        if m.status == 2:  # GRB.OPTIMAL
            objective_value = m.ObjVal  # Number of changed reactions
            flux_distribution = {}
            for reaction in model_reactions:
                flux_distribution[reaction] = float(v[reaction].x)
                # Round near-zero values to exactly zero
                if abs(float(v[reaction].x)) <= 1e-6:
                    flux_distribution[reaction] = 0.0

            return m.status, objective_value, flux_distribution
        else:
            return m.status, False, False

    def run_FBA(self, new_objective='', flux_constraints={}, inf_flag=False, 
                internal_flux_minimization=False, mode='max'):
        """
        Flux Balance Analysis (FBA).
        
        FBA maximizes or minimizes an objective function (typically biomass or a target
        metabolite production) subject to stoichiometric constraints and flux bounds.
        
        Parameters:
        -----------
        new_objective : str, optional
            Reaction ID to use as objective function. If empty, uses model's default objective.
        flux_constraints : dict
            Additional flux constraints. Keys are reaction IDs, values are tuples
            of (lower_bound, upper_bound).
        inf_flag : bool, optional (default=False)
            If False, replaces infinite bounds with ±1000.
        internal_flux_minimization : bool, optional (default=False)
            If True, performs parsimonious FBA (pFBA) to minimize total flux while
            maintaining optimal objective value.
        mode : str, optional (default='max')
            Optimization mode: 'max' for maximization, 'min' for minimization.
            
        Returns:
        --------
        tuple: (status, objective_value, flux_distribution)
            - status: Gurobi optimization status (2 = optimal)
            - objective_value: Optimized objective function value (or total flux for pFBA)
            - flux_distribution: Dict of reaction IDs to flux values
            
        Notes:
        ------
        Parsimonious FBA (pFBA) is useful for obtaining more biologically realistic
        flux distributions by minimizing the total sum of fluxes while maintaining
        optimal growth or production.
        """
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions

        # Determine objective reaction
        if new_objective == '':
            objective = self.objective
        else:
            objective = new_objective

        Smatrix = self.Smatrix
        lower_boundary_constraints = self.lower_boundary_constraints.copy()
        upper_boundary_constraints = self.upper_boundary_constraints.copy()

        # Replace infinite bounds with finite values if inf_flag is False
        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        # Prepare Gurobi data structures
        pairs, coffvalue = multidict(Smatrix)
        pairs = tuplelist(pairs)

        # Create Gurobi model
        m = Model('FBA')
        m.setParam('OutputFlag', 0)
        m.reset()

        # Create variables
        v = {}  # Flux variables
        fplus = {}  # Positive flux (for pFBA)
        fminus = {}  # Negative flux (for pFBA)

        m.update()

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], 
                                            name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            # Auxiliary variables for pFBA
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f"fplus_{each_reaction}")
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f"fminus_{each_reaction}")
        
        m.update()

        # Add steady-state mass balance constraints (Sv = 0)
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                v[reaction] * coffvalue[metabolite, reaction] 
                for metabolite, reaction in pairs.select(each_metabolite, '*')) == 0,
                name=f"mass_balance_{each_metabolite}")

        m.update()
        
        # Set primary objective (maximize or minimize target reaction flux)
        if mode == 'max':
            m.setObjective(v[objective], GRB.MAXIMIZE)
        elif mode == 'min':
            m.setObjective(v[objective], GRB.MINIMIZE)

        # Solve primary optimization
        m.optimize()
        
        if m.status == 2:  # GRB.OPTIMAL
            objective_value = m.ObjVal
            
            # Parsimonious FBA: minimize total flux while maintaining optimal objective
            if internal_flux_minimization:
                # Fix objective flux to optimal value
                m.addConstr(v[objective] == objective_value, name="fix_objective")

                # Add flux decomposition constraints for all reactions
                for each_reaction in model_reactions:
                    m.addConstr(fplus[each_reaction] - fminus[each_reaction] == v[each_reaction],
                               name=f"flux_decomp_{each_reaction}")

                m.update()
                
                # New objective: minimize sum of absolute fluxes
                m.setObjective(
                    quicksum((fplus[each_reaction] + fminus[each_reaction]) 
                            for each_reaction in model_reactions),
                    GRB.MINIMIZE)
                
                # Solve secondary optimization
                m.optimize()
                
                if m.status == 2:  # GRB.OPTIMAL
                    objective_value = m.ObjVal  # Total flux sum
                    flux_distribution = {}
                    for reaction in model_reactions:
                        flux_distribution[reaction] = float(v[reaction].x)
                        # Round near-zero values to exactly zero
                        if abs(float(v[reaction].x)) <= 1e-6:
                            flux_distribution[reaction] = 0.0
                    return m.status, objective_value, flux_distribution
                else:
                    return m.status, False, False
            else:
                # Standard FBA: return optimal flux distribution
                flux_distribution = {}
                for reaction in model_reactions:
                    flux_distribution[reaction] = float(v[reaction].x)
                    # Round near-zero values to exactly zero
                    if abs(float(v[reaction].x)) <= 1e-6:
                        flux_distribution[reaction] = 0.0
                return m.status, objective_value, flux_distribution
        
        return m.status, False, False

    def read_model(self, filename):
        """
        Load a metabolic model from an SBML file.
        
        Parameters:
        -----------
        filename : str
            Path to the SBML model file.
            
        Returns:
        --------
        tuple: (metabolites, reactions, Smatrix, lb, ub, objective)
            Components of the loaded model.
        """
        model = io.read_sbml_model(filename)
        return self.load_cobra_model(model)

    def load_cobra_model(self, cobra_model):
        """
        Load a COBRApy model into the simulator.
        
        Extracts all necessary components from a COBRApy model including metabolites,
        reactions, stoichiometric matrix, flux bounds, and objective function.
        
        Parameters:
        -----------
        cobra_model : cobra.Model
            A COBRApy model object.
            
        Returns:
        --------
        tuple: (metabolites, reactions, Smatrix, lb, ub, objective)
            - metabolites: List of metabolite IDs
            - reactions: List of reaction IDs
            - Smatrix: Dict mapping (metabolite, reaction) to stoichiometric coefficient
            - lb: Dict of lower bounds for each reaction
            - ub: Dict of upper bounds for each reaction
            - objective: Reaction ID of the objective function
        """
        self.cobra_model = cobra_model
        model = cobra_model
        model_metabolites = []
        model_reactions = []
        model_genes = []
        lower_boundary_constraints = {}
        upper_boundary_constraints = {}
        objective_reaction = ''
        
        # Extract metabolites
        for each_metabolite in model.metabolites:
            model_metabolites.append(each_metabolite.id)

        # Extract genes
        model_genes = [each_gene.id for each_gene in model.genes]

        # Build stoichiometric matrix
        Smatrix = {}

        for each_reaction in model.reactions:
            # Identify objective reaction
            if each_reaction.objective_coefficient == 1.0:
                objective_reaction = each_reaction.id

            # Extract stoichiometric coefficients for reactants
            reactant_list = each_reaction.reactants
            reactant_coff_list = each_reaction.get_coefficients(reactant_list)
            
            # Extract stoichiometric coefficients for products
            product_list = each_reaction.products
            product_coff_list = each_reaction.get_coefficients(product_list)
            
            reactant_coff_list = list(reactant_coff_list)
            product_coff_list = list(product_coff_list)
            
            # Add reactant coefficients to Smatrix (negative by convention)
            for i in range(len(reactant_list)):
                Smatrix[(reactant_list[i].id, each_reaction.id)] = reactant_coff_list[i]

            # Add product coefficients to Smatrix (positive by convention)
            for i in range(len(product_list)):
                Smatrix[(product_list[i].id, each_reaction.id)] = product_coff_list[i]

            # Store reaction ID
            model_reactions.append(each_reaction.id)
            
            # Extract flux bounds
            lb = each_reaction.lower_bound
            ub = each_reaction.upper_bound
            
            # Convert very large bounds to infinity for cleaner representation
            if lb < -1000.0:
                lb = float('-inf')
            if ub > 1000.0:
                ub = float('inf')
            
            lower_boundary_constraints[each_reaction.id] = lb
            upper_boundary_constraints[each_reaction.id] = ub

        # Store all model components as instance variables
        self.model_metabolites = model_metabolites
        self.model_reactions = model_reactions
        self.model_genes = model_genes
        self.Smatrix = Smatrix
        self.lower_boundary_constraints = lower_boundary_constraints
        self.upper_boundary_constraints = upper_boundary_constraints
        self.objective = objective_reaction

        return (model_metabolites, model_reactions, Smatrix, lower_boundary_constraints, 
                upper_boundary_constraints, objective_reaction)


