import cvxpy as cp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from quantfolio.constraints import Constraints
from quantfolio.universe import Universe


class Optimizer:
    def __init__(self, universe: Universe, constraints: Constraints, solver: str = "cvxpy", sense: str = "maximize", max_assets=5):
        self.universe = universe
        self.constraints = constraints
        self.solver = solver
        self.sense = self._parse_sense(sense)
        self.max_assets = max_assets

    def _parse_sense(self, sense: str):
        sense_map = {"maximize": 1, "minimize": -1}
        if sense not in sense_map:
            raise ValueError(
                "Invalid sense value. Choose 'maximize' or 'minimize'.")
        return sense_map[sense]

    def optimize(self):
        df = self.universe.get_data()
        ids = df[self.universe.id_column].tolist()
        values = df["value"].values

        if self.solver == "cvxpy":
            # Define decision variables
            x = cp.Variable(len(ids), boolean=True)

            # Define objective
            objective = cp.Maximize(self.sense * values @ x)

            # Define constraints
            constraints = self.constraints.apply_constraints(
                x, self.universe, self.max_assets, solver=self.solver)

            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Get results
            selected_ids = [ids[i]
                            for i in range(len(ids)) if x.value[i] > 0.5]

        elif self.solver == "pyomo":
            model = pyo.ConcreteModel()
            model.x = pyo.Var(range(len(ids)), domain=pyo.Binary)

            model.objective = pyo.Objective(expr=self.sense * sum(values[i] * model.x[i] for i in range(
                len(ids))), sense=pyo.maximize if self.sense == 1 else pyo.minimize)

            model.constraints = pyo.ConstraintList()
            for c in self.constraints.apply_constraints([model.x[i] for i in range(len(ids))], self.universe, self.max_assets, solver=self.solver):
                model.constraints.add(c)

            solver = SolverFactory("glpk")
            solver.solve(model)

            # Get results
            selected_ids = [ids[i] for i in range(
                len(ids)) if pyo.value(model.x[i]) > 0.5]

        return selected_ids
