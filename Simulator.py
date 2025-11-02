import cobra.io as io
from optlang import Model, Variable, Constraint, Objective


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

        # Create optlang model
        m = Model(name='MOMA')

        # Create variables
        v = {}  # Flux variables
        fplus = {}  # Positive deviation from wild-type
        fminus = {}  # Negative deviation from wild-type

        variables_to_add = []

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=flux_constraints[each_reaction][0],
                    ub=flux_constraints[each_reaction][1]
                )
            else:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=lower_boundary_constraints[each_reaction],
                    ub=upper_boundary_constraints[each_reaction]
                )
            # Deviation variables (non-negative)
            fplus[each_reaction] = Variable(f"fplus_{each_reaction}", lb=0.0, ub=1000.0)
            fminus[each_reaction] = Variable(f"fminus_{each_reaction}", lb=0.0, ub=1000.0)

            variables_to_add.extend([v[each_reaction], fplus[each_reaction], fminus[each_reaction]])

        m.add(variables_to_add)

        # Add constraints relating flux to deviations
        constraints = []

        for each_reaction in model_reactions:
            # v = fplus - fminus (flux decomposition)
            constraints.append(Constraint(
                v[each_reaction] - fplus[each_reaction] + fminus[each_reaction],
                lb=0, ub=0,
                name=f"flux_decomp_{each_reaction}"
            ))

            # fplus >= v - wild_flux (captures positive deviation)
            if each_reaction in wild_flux:
                constraints.append(Constraint(
                    fplus[each_reaction] - v[each_reaction] + wild_flux[each_reaction],
                    lb=0,
                    name=f"fplus_lb_{each_reaction}"
                ))
                # fminus >= wild_flux - v (captures negative deviation)
                constraints.append(Constraint(
                    fminus[each_reaction] + v[each_reaction] - wild_flux[each_reaction],
                    lb=0,
                    name=f"fminus_lb_{each_reaction}"
                ))

        m.add(constraints)

        # Add steady-state mass balance constraints (Sv = 0)
        mass_balance_constraints = []

        for each_metabolite in model_metabolites:
            # Find all reactions involving this metabolite
            metabolite_reactions = [(met, rxn) for (met, rxn) in Smatrix.keys() if met == each_metabolite]

            if len(metabolite_reactions) == 0:
                continue

            # Create mass balance expression: sum(S_ij * v_j) = 0
            expr = sum(v[reaction] * Smatrix[metabolite, reaction]
                      for metabolite, reaction in metabolite_reactions)

            mass_balance_constraints.append(Constraint(
                expr, lb=0, ub=0,
                name=f"mass_balance_{each_metabolite}"
            ))

        m.add(mass_balance_constraints)

        # Set objective: minimize sum of absolute deviations
        # This approximates Euclidean distance minimization
        target_reactions = wild_flux.keys()
        objective_expr = sum((fplus[each_reaction] + fminus[each_reaction])
                            for each_reaction in target_reactions)
        m.objective = Objective(objective_expr, direction='min')

        # Solve the optimization problem
        m.optimize()

        # Extract results if optimization was successful
        if m.status == 'optimal':
            flux_distribution = {}
            for reaction in model_reactions:
                flux_distribution[reaction] = float(v[reaction].primal)
                # Round near-zero values to exactly zero
                if abs(float(v[reaction].primal)) <= 1e-6:
                    flux_distribution[reaction] = 0.0

            return m.status, m.objective.value, flux_distribution
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

        # Create optlang model
        m = Model(name='ROOM')

        # Create variables
        v = {}  # Flux variables
        y = {}  # Binary variables: 1 if reaction flux significantly changed, 0 otherwise

        variables_to_add = []

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=flux_constraints[each_reaction][0],
                    ub=flux_constraints[each_reaction][1]
                )
            else:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=lower_boundary_constraints[each_reaction],
                    ub=upper_boundary_constraints[each_reaction]
                )

            # Binary indicator variable for significant flux change
            y[each_reaction] = Variable(f"y_{each_reaction}", type='binary')

            variables_to_add.extend([v[each_reaction], y[each_reaction]])

        m.add(variables_to_add)

        # Add steady-state mass balance constraints (Sv = 0)
        mass_balance_constraints = []

        for each_metabolite in model_metabolites:
            # Find all reactions involving this metabolite
            metabolite_reactions = [(met, rxn) for (met, rxn) in Smatrix.keys() if met == each_metabolite]

            if len(metabolite_reactions) == 0:
                continue

            # Create mass balance expression: sum(S_ij * v_j) = 0
            expr = sum(v[reaction] * Smatrix[metabolite, reaction]
                      for metabolite, reaction in metabolite_reactions)

            mass_balance_constraints.append(Constraint(
                expr, lb=0, ub=0,
                name=f"mass_balance_{each_metabolite}"
            ))

        m.add(mass_balance_constraints)

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
            m.add(Constraint(
                v[each_reaction] - M_upper * y[each_reaction],
                ub=w_upper,
                name=f"room_upper_{each_reaction}"
            ))

            # v >= w_lower - M_lower * y
            # When y=0: v >= w_lower (enforces lower bound of allowable range)
            # When y=1: v >= w_lower - M_lower (effectively no constraint)
            m.add(Constraint(
                v[each_reaction] + M_lower * y[each_reaction],
                lb=w_lower,
                name=f"room_lower_{each_reaction}"
            ))

        # Set objective: minimize number of significantly changed reactions
        objective_expr = sum(y[each_reaction] for each_reaction in wild_flux.keys())
        m.objective = Objective(objective_expr, direction='min')

        # Solve the optimization problem
        m.optimize()

        # Extract results if optimization was successful
        if m.status == 'optimal':
            objective_value = m.objective.value  # Number of changed reactions
            flux_distribution = {}
            for reaction in model_reactions:
                flux_distribution[reaction] = float(v[reaction].primal)
                # Round near-zero values to exactly zero
                if abs(float(v[reaction].primal)) <= 1e-6:
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

        # Create optlang model
        m = Model(name='FBA')

        # Create variables
        v = {}  # Flux variables
        fplus = {}  # Positive flux (for pFBA)
        fminus = {}  # Negative flux (for pFBA)

        variables_to_add = []

        for each_reaction in model_reactions:
            # Set flux bounds based on constraints or model defaults
            if each_reaction in flux_constraints:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=flux_constraints[each_reaction][0],
                    ub=flux_constraints[each_reaction][1]
                )
            else:
                v[each_reaction] = Variable(
                    each_reaction,
                    lb=lower_boundary_constraints[each_reaction],
                    ub=upper_boundary_constraints[each_reaction]
                )
            # Auxiliary variables for pFBA
            fplus[each_reaction] = Variable(f"fplus_{each_reaction}", lb=0.0, ub=1000.0)
            fminus[each_reaction] = Variable(f"fminus_{each_reaction}", lb=0.0, ub=1000.0)

            variables_to_add.extend([v[each_reaction], fplus[each_reaction], fminus[each_reaction]])

        m.add(variables_to_add)

        # Add steady-state mass balance constraints (Sv = 0)
        mass_balance_constraints = []

        for each_metabolite in model_metabolites:
            # Find all reactions involving this metabolite
            metabolite_reactions = [(met, rxn) for (met, rxn) in Smatrix.keys() if met == each_metabolite]

            if len(metabolite_reactions) == 0:
                continue

            # Create mass balance expression: sum(S_ij * v_j) = 0
            expr = sum(v[reaction] * Smatrix[metabolite, reaction]
                      for metabolite, reaction in metabolite_reactions)

            mass_balance_constraints.append(Constraint(
                expr, lb=0, ub=0,
                name=f"mass_balance_{each_metabolite}"
            ))

        m.add(mass_balance_constraints)

        # Set primary objective (maximize or minimize target reaction flux)
        if mode == 'max':
            m.objective = Objective(v[objective], direction='max')
        elif mode == 'min':
            m.objective = Objective(v[objective], direction='min')

        # Solve primary optimization
        m.optimize()

        if m.status == 'optimal':
            objective_value = m.objective.value

            # Parsimonious FBA: minimize total flux while maintaining optimal objective
            if internal_flux_minimization:
                # Fix objective flux to optimal value
                m.add(Constraint(v[objective], lb=objective_value, ub=objective_value, name="fix_objective"))

                # Add flux decomposition constraints for all reactions
                pfba_constraints = []
                for each_reaction in model_reactions:
                    pfba_constraints.append(Constraint(
                        fplus[each_reaction] - fminus[each_reaction] - v[each_reaction],
                        lb=0, ub=0,
                        name=f"flux_decomp_{each_reaction}"
                    ))

                m.add(pfba_constraints)

                # New objective: minimize sum of absolute fluxes
                objective_expr = sum((fplus[each_reaction] + fminus[each_reaction])
                                    for each_reaction in model_reactions)
                m.objective = Objective(objective_expr, direction='min')

                # Solve secondary optimization
                m.optimize()

                if m.status == 'optimal':
                    objective_value = m.objective.value  # Total flux sum
                    flux_distribution = {}
                    for reaction in model_reactions:
                        flux_distribution[reaction] = float(v[reaction].primal)
                        # Round near-zero values to exactly zero
                        if abs(float(v[reaction].primal)) <= 1e-6:
                            flux_distribution[reaction] = 0.0
                    return m.status, objective_value, flux_distribution
                else:
                    return m.status, False, False
            else:
                # Standard FBA: return optimal flux distribution
                flux_distribution = {}
                for reaction in model_reactions:
                    flux_distribution[reaction] = float(v[reaction].primal)
                    # Round near-zero values to exactly zero
                    if abs(float(v[reaction].primal)) <= 1e-6:
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


