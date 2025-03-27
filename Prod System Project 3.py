import gurobipy as gp
import numpy as np

# ----- PART 1: Aggregate Production Plan -----
demands = [150, 300, 240]
hours_per_unit = 3
reg_hours = 300
max_ot = 0.5 * reg_hours
cost = {"material": 15, "regular": 12, "overtime": 18, "inventory": 15}
init_inv = 20
nq = 3  # Number of quarters

m = gp.Model("Aggregate_Production")
m.setParam("OutputFlag", 0)
m.setParam("Threads", 16)

# Create decision variables for production, inventory, overtime per quarter
prod = [m.addVar(lb=0, name=f"P{i+1}") for i in range(nq)]
inv  = [m.addVar(lb=0, name=f"I{i+1}") for i in range(nq)]
ot   = [m.addVar(lb=0, name=f"OT{i+1}") for i in range(nq)]

# Inventory balance constraints
m.addConstr(init_inv + prod[0] - demands[0] == inv[0])
for i in range(1, nq):
    m.addConstr(inv[i-1] + prod[i] - demands[i] == inv[i])

# Labor and overtime constraints
for i in range(nq):
    m.addConstr(hours_per_unit * prod[i] <= reg_hours + ot[i])
    m.addConstr(ot[i] <= max_ot)

# Objective: sum over quarters (note: fixed reg_hours cost is added each quarter)
m.setObjective(gp.quicksum(cost["material"]*prod[i] + cost["regular"]*reg_hours +
                            cost["overtime"]*ot[i] + cost["inventory"]*inv[i] for i in range(nq)),
              gp.GRB.MINIMIZE)
m.optimize()
if m.status != gp.GRB.OPTIMAL:
    m.feasRelax(0, False, None, None, None, None, None)
    m.optimize()
if m.status != gp.GRB.OPTIMAL:
    print("Aggregate model infeasible. Using fallback plan.")
    agg_prod = demands.copy()
    agg_inv = [0, 0, 0]
else:
    agg_prod = [p.X for p in prod]
    agg_inv  = [v.X for v in inv]
print("Aggregate Plan:")
for i in range(nq):
    print(f" Q{i+1}: Prod = {agg_prod[i]:.2f}, Inv = {agg_inv[i]:.2f}")

# ----- PART 2: Disaggregate Plan -----
# Proportions: A:20%, B:20%, C:60%
def disagg(val): return [0.2*val, 0.2*val, 0.6*val]
prodA = [disagg(agg_prod[i])[0] for i in range(nq)]
prodB = [disagg(agg_prod[i])[1] for i in range(nq)]
prodC = [disagg(agg_prod[i])[2] for i in range(nq)]
print("\nDisaggregate Plan:")
for i in range(nq):
    print(f" Q{i+1}: A = {prodA[i]:.2f}, B = {prodB[i]:.2f}, C = {prodC[i]:.2f}")

# ----- PART 3: Master Production Schedule (MPS) -----
# 9 periods (3 per quarter)
periods = 9
mpsA, mpsB, mpsC = [], [], []
for i in range(nq):
    for _ in range(3):
        mpsA.append(prodA[i] / 3)
        mpsB.append(prodB[i] / 3)
        mpsC.append(prodC[i] / 3)
print("\nMPS (per period):")
print("Period\tA\tB\tC")
for i in range(periods):
    print(f"{i+1}\t{mpsA[i]:.2f}\t{mpsB[i]:.2f}\t{mpsC[i]:.2f}")

# ----- PART 4: Materials Requirement Plan (MRP) -----
# Parts: Blueberry = A + C, Pear = A, Strawberry = B + C, Banana = B + C
init_part = 50
parts = ["Blueberry", "Pear", "Strawberry", "Banana"]
mrp = {p: {"Gross": [0]*periods, "OnHand": [0]*periods, "Net": [0]*periods, "Planned": [0]*periods} for p in parts}
for i in range(periods):
    a, b, c = mpsA[i], mpsB[i], mpsC[i]
    mrp["Blueberry"]["Gross"][i] = a + c
    mrp["Pear"]["Gross"][i] = a
    mrp["Strawberry"]["Gross"][i] = b + c
    mrp["Banana"]["Gross"][i] = b + c
for p in parts:
    onh = init_part
    for i in range(periods):
        gr = mrp[p]["Gross"][i]
        net = max(gr - onh, 0)
        mrp[p]["Net"][i] = net
        mrp[p]["Planned"][i] = net
        onh = onh + net - gr
        mrp[p]["OnHand"][i] = onh
print("\nMRP:")
for p in parts:
    print(f"\nPart: {p}")
    print("Per.\tGross\tOnHand\tNet\tPlanned")
    for i in range(periods):
        print(f"{i+1}\t{mrp[p]['Gross'][i]:.2f}\t{mrp[p]['OnHand'][i]:.2f}\t{mrp[p]['Net'][i]:.2f}\t{mrp[p]['Planned'][i]:.2f}")

# ----- PART 5: Simulation (100,000 iterations) -----
num_sim = 100000
d_means = np.array([
    [10,10,30], [10,10,30], [10,10,30],
    [20,20,60], [20,20,60], [20,20,60],
    [16,16,48], [16,16,48], [16,16,48]
], dtype=float)
d_stds = np.array([
    [3,3,6], [3,3,6], [3,3,6],
    [6,6,12], [6,6,12], [6,6,12],
    [4,4,8], [4,4,8], [4,4,8]
], dtype=float)
prod_sched = np.array([mpsA, mpsB, mpsC]).T
init_inv_arr = np.array([0.2*init_inv, 0.2*init_inv, 0.6*init_inv])
selling = np.array([100,130,115])
prod_cost = np.array([10,20,15])
hold_cost = np.array([4.5,5.5,5.0])
fixed_cost = np.sum(prod_sched * prod_cost)

rev = np.zeros(num_sim)
hold_total = np.zeros(num_sim)
lost_total = np.zeros(num_sim)
inv_arr = np.tile(init_inv_arr, (num_sim, 1))
for p in range(periods):
    d = np.maximum(np.round(np.random.normal(d_means[p], d_stds[p], (num_sim, 3))), 0)
    avail = inv_arr + prod_sched[p]
    sales = np.minimum(avail, d)
    lost = d - sales
    rev += np.sum(sales * selling, axis=1)
    end_inv = avail - sales
    hold_total += np.sum(end_inv * hold_cost, axis=1)
    inv_arr = end_inv
    lost_total += np.sum(lost, axis=1)
profit = rev - fixed_cost - hold_total
print("\nSimulation Results (100k iterations):")
print(f"Avg Rev: ${np.mean(rev):.2f}, Fixed Cost: ${fixed_cost:.2f}, Avg Hold: ${np.mean(hold_total):.2f}, Avg Lost: {np.mean(lost_total):.2f}, Avg Profit: ${np.mean(profit):.2f}")
