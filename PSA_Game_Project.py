import gurobipy as gp
from gurobipy import GRB
import numpy as np

##################
### Parameters ###
##################

# Dimensions: Time, Goods, N-scenarios

T = [i for i in range(9)] # T = periods
Q = [q for q in range(3)] # Q = quarters

goods = [i for i in range(3)] # g = goods
raws = [i for i in range(4)] # r = raw materials

num_scenarios = 1000 # for now
N = [n for n in range(num_scenarios)]

# D[t, g, n] = Demand for good g at time t in scenario n
d_means = np.array([
    [10,10,30], [10,10,30], [10,10,30],
    [20,20,60], [20,20,60], [20,20,60],
    [16,16,48], [16,16,48], [16,16,48]
])

d_stds = np.array([
    [3,3,6], [3,3,6], [3,3,6],
    [6,6,12], [6,6,12], [6,6,12],
    [4,4,8], [4,4,8], [4,4,8]
])

# D[t, g, n] = demand for at time t for good g in scenario n
D = np.array([[np.random.normal(d_means[t, g], d_stds[t, g], num_scenarios) for g in goods] for t in T])
    # Double checked all the dimensions, the above comment is correct
    
# Remove all instances of negative demand
for t in T:
    for g in goods:
        for n in N:
            if D[t, g, n] < 0:
                D[t, g, n] = 0
                
                
labor_requirements = [2, 4, 3] # A, B, C

raw_lead_times = [1, 1, 2, 1] # B, G, R, Y

initial_hours = 300 # Given in problem

expected_quality = [1, 1, 0.95, 0.9] # For ordering raw materials

## Costs
c_X = 130
c_Y = -130
c_S = [-100, -130, -115]
c_I = [4.5, 5.5, 5]
c_R = [5, 5, 15, 10]
c_Ir = [1.5, 1.5, 3.1, 2.2]
c_L = 12
c_O = 18
c_I_init = [30, 60, 45]
c_Ir_init = [5+1.5, 5+1.5, 15+3.1, 10+2.2]
c_Z = 75
c_F = 8
c_H = 6

#############
### Model ###
#############

m = gp.Model("Aggregate_Production")

## Decision Variables
    # All the inputs to the website are independent of the scenario
        # Production, labor, raw material purchasing, etc.
        
    # All other variables are dependent on the scenario
        # Inventory, Sales, etc.
        
# Deterministic Decision Variables
P = m.addVars(T, goods, lb = 0, vtype=GRB.INTEGER) # Production
X = m.addVars(T, goods, lb = 0, vtype=GRB.BINARY) # Production Set-ups
Y = m.addVars(T[1:], lb = 0, vtype=GRB.BINARY) # Set-up carry-over
O = m.addVars(T, lb = 0, vtype = GRB.INTEGER) # Overtime hours
L = m.addVars(T, lb = 0, vtype = GRB.INTEGER) # Labor Hours
L_quarter = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Labor available for each quarter
aux = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Auxiliary variable for later use
H = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Hiring for each quarter
F = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Firing for each quarter


U = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Use raw materials
R = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Purchasing raw materials
Z = m.addVars(T, raws, lb = 0, vtype=GRB.BINARY)  # Order required 

I_init = m.addVars(goods, lb = 0, ub = 20, vtype = GRB.INTEGER) # Initial inventory
    # Note: We are only allowed up to 20 units in our starting inventory

I_r = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Inventory of raw materials    
I_r_init = m.addVars(raws, lb = 0, ub = 50, vtype=GRB.INTEGER) # Initial raw inventory
    # Note: we are only allowed up to 50 units in our starting inventory

# Scenario-Dependent Decision Variables
S = m.addVars(T, goods, N, lb = 0, vtype=GRB.INTEGER) # Sales
I = m.addVars(T, goods, N, lb = 0, vtype=GRB.INTEGER) # Inventory


## Inventory balance constraints

# Initial
m.addConstrs(I_init[g] + P[0, g] - S[0, g, n] == I[0, g, n] for g in goods for n in N)

# Balance
m.addConstrs(I[t, g, n] == I[t-1, g, n] + P[t-1, g] - S[0, g, n] for g in goods for t in T[1:] for n in N)

# Sales can never be higher than demand
m.addConstrs(S[t, g, n] <= D[t, g, n] for t in T for g in goods for n in N)
    # This allows profits to be tied to the variations in demand


## Labor and overtime constraints

# Labor in each period must sum to the labor available for the quarter
m.addConstrs(gp.quicksum(L[t] for t in range((q+1)*3-3, (q+1)*3)) == L_quarter[q] for q in Q)

# Labor by period - including the CDC will be annoying
# TDL

# Labor in the quarter must be a multiple of three
m.addConstrs(L_quarter[q] == 3 * aux[q] for q in Q)

# Hiring and Firing
m.addConstr(L_quarter[0] == 300 + H[0] - F[0]) # Initial
m.addConstrs(L_quarter[q] == L_quarter[q-1] + H[q-1] - F[q-1] for q in Q[1:]) # Rest of the quarters

# Overtime
m.addConstrs(O[t] <= 1.5 * L[t] for t in T)

# Labor limiting Production
m.addConstrs(L[t] + O[t] >= gp.quicksum(P[t, g] * labor_requirements[g] for g in goods) for t in T)

## Raw Material Requirements

# Initial
m.addConstrs(I_r_init[r] + expected_quality[r] * R[0, r] - U[0, r] == I_r[0, r] for r in [0, 1, 3])

# Balance - All the parts with one week lead time are easy
m.addConstrs(I_r[t,r] == I_r[t-1, r] + expected_quality[r] * R[t-1, r] - U[t, r] for r in [0, 1, 3] for t in T[1:])

# Strawberries - lead time is 2 weeks, which makes life a little more tricky
m.addConstr(I_r[0, 2] == I_r_init[2] - U[0, 2]) # First Time period
m.addConstr(I_r[1, 2] == I_r[0, 2] - U[1, 2]) # Second time period
m.addConstrs(I_r[t, 2] == I_r[t-1, 2] - U[t, 2] + expected_quality[2] * R[t-2, 2] for t in T[2:]) # Rest of the periods

# Raw Materials limiting Production
m.addConstrs(P[t, 0] + P[t, 2] <= U[t, 0] for t in T) # Blueberries  used for A and C
m.addConstrs(P[t, 0] + P[t, 1] <= U[t, 1] for t in T) #    Pears     used for A and B
m.addConstrs(P[t, 1] <= U[t, 2] for t in T)           # Strawberries used for B
m.addConstrs(P[t, 2] <= U[t, 3] for t in T)           #   Bananas    used for C

## Set Ups

# First, determine when set-ups are required
m.addConstrs(P[t, g] <= 1000 * X[t, g] for g in goods for t in T)

# Second, determine if a set-up can be carried over
m.addConstrs(Y[t] <= gp.quicksum(X[t, g] + X[t-1, g] for g in goods) for t in T[1:])

# Raw material orders
m.addConstrs(R[t, r] <= 1000 * Z[t, r] for r in raws for t in T)




## Objective Function
m.setObjective(gp.quicksum((1/num_scenarios) * c_S[g] * S[t, g, n] for g in goods for t in T for n in N)
             + gp.quicksum(c_X * X[t, g] for g in goods for t in T)
             + gp.quicksum(c_Y * Y[t] for t in T[1:])
             + gp.quicksum((1/num_scenarios) * c_I[g] * I[t, g, n] for g in goods for t in T for n in N)
             + gp.quicksum(c_Ir[r] * I_r[t, r] for r in raws for t in T)
             + gp.quicksum(c_L * L[t] for t in T)
             + gp.quicksum(c_L * L[t] for t in T)
             + gp.quicksum(c_I_init[g] * I_init[g] for g in goods)
             + gp.quicksum(c_Ir_init[r] * I_r_init[r] for r in raws)
             + gp.quicksum(c_Z * Z[t, r] for t in T for r in raws)
             + gp.quicksum(c_H * H[q] for q in Q)
             + gp.quicksum(c_F * F[q] for q in Q)
             )


m.optimize()


# TDL:
    # Labor Shortage - Diran or Ben
    # Objective function - Ben

## Everything below here is untouched ##

"""
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
"""
print()





