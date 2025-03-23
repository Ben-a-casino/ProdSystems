import gurobipy as gp
from gurobipy import GRB

# Create model
m = gp.Model("Aggregate_Production_Plan")

# Decision variables
prod_Q1 = m.addVar(lb=0, name="Production_Q1")
prod_Q2 = m.addVar(lb=0, name="Production_Q2")
prod_Q3 = m.addVar(lb=0, name="Production_Q3")

inv_Q1 = m.addVar(lb=0, name="Inventory_Q1")
inv_Q2 = m.addVar(lb=0, name="Inventory_Q2")
inv_Q3 = m.addVar(lb=0, name="Inventory_Q3")

ot_Q1   = m.addVar(lb=0, name="Overtime_Q1")
ot_Q2   = m.addVar(lb=0, name="Overtime_Q2")
ot_Q3   = m.addVar(lb=0, name="Overtime_Q3")

# Inventory balance constraints
m.addConstr(inv_Q1 == prod_Q1 - 150,        "InventoryBalance_Q1")
m.addConstr(inv_Q2 == inv_Q1 + prod_Q2 - 300, "InventoryBalance_Q2")
m.addConstr(inv_Q3 == inv_Q2 + prod_Q3 - 240, "InventoryBalance_Q3")

# Labor capacity 
m.addConstr(3 * prod_Q1 - ot_Q1 <= 300, "RegularCapacity_Q1")
m.addConstr(3 * prod_Q2 - ot_Q2 <= 300, "RegularCapacity_Q2")
m.addConstr(3 * prod_Q3 - ot_Q3 <= 300, "RegularCapacity_Q3")

# Overtime limit (50% of regular hours)
m.addConstr(ot_Q1 <= 150, "OvertimeLimit_Q1")
m.addConstr(ot_Q2 <= 150, "OvertimeLimit_Q2")
m.addConstr(ot_Q3 <= 150, "OvertimeLimit_Q3")

# Mimimize function
m.setObjective(
    15 * prod_Q1 + 36 * prod_Q1 + 18 * ot_Q1 + 15 * inv_Q1 +
    15 * prod_Q2 + 36 * prod_Q2 + 18 * ot_Q2 + 15 * inv_Q2 +
    15 * prod_Q3 + 36 * prod_Q3 + 18 * ot_Q3 + 15 * inv_Q3,
    GRB.MINIMIZE
)

m.optimize()

print("Optimal Plan:")
print(f" Q1 – Production: {prod_Q1.X:.2f}, Inventory: {inv_Q1.X:.2f}, Overtime: {ot_Q1.X:.2f}")
print(f" Q2 – Production: {prod_Q2.X:.2f}, Inventory: {inv_Q2.X:.2f}, Overtime: {ot_Q2.X:.2f}")
print(f" Q3 – Production: {prod_Q3.X:.2f}, Inventory: {inv_Q3.X:.2f}, Overtime: {ot_Q3.X:.2f}")
print(f"Total Cost = ${m.ObjVal:.2f}")
