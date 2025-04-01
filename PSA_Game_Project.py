import gurobipy as gp
from gurobipy import GRB
import numpy as np
from beautifultable import BeautifulTable

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

## Simulating the pandemic

# Goal: Construct the following variable:
    # sick[t, n] = The fraction of workers absent at time t in scenario n
# In a given scenario, only one of the periods should have absentees
    # Specifically, one of the last three periods
# Up to 50% absent

# First, determine which of the three periods will have the pandemic
pandemic = np.random.randint(T[-3], T[-1], num_scenarios)

# Second, create the variable but assign all zeros
sick = np.zeros((9, num_scenarios))

# Recall np.random.rand() produces a random number between 0 and 1
    # Therefore 0.5 * np.random.rand() produces a random number between 0 and 0.5
    
# Finally, replace a zero with absentees if the pandemic hits
for n in N:
    sick[pandemic[n], n] = np.random.rand() * 0.5

    
## Costs
c_X = 130
c_Z = -130
c_S = np.array([-100, -130, -115])
c_I = [4.5, 5.5, 5]
c_R = [5, 5, 15, 10]
c_Ir = [1.5, 1.5, 3.1, 2.2]
c_L = 12
c_O = 18
c_I_init = [30, 60, 45]
c_Ir_init = [5+1.5, 5+1.5, 15+3.1, 10+2.2]
c_V = 75
c_F = 8
c_H = 6

#############
### Model ###
#############

m = gp.Model("Aggregate_Production")
m.setParam("MIPGap", 0.0005)
## Decision Variables
    # All the inputs to the website are independent of the scenario
        # Production, labor, raw material purchasing, etc.
        
    # All other variables are dependent on the scenario
        # Inventory, Sales, etc.
        
# Deterministic Decision Variables
P = m.addVars(T, goods, lb = 0, vtype=GRB.INTEGER) # Production
X = m.addVars(T, goods, lb = 0, vtype=GRB.BINARY) # Production Set-ups
Y = m.addVars(T[1:], goods, lb = 0, vtype=GRB.BINARY) # Set-up carry-over
Z = m.addVars(T[1:], lb=0, vtype=GRB.BINARY) # Max Set-up saves
L = m.addVars(T, lb = 0, vtype = GRB.INTEGER) # Labor Hours 
O = m.addVars(T, lb = 0, vtype = GRB.INTEGER) # Overtime hours
L_quarter = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Labor available for each quarter
aux = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Auxiliary variable for later use
H = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Hiring for each quarter
F = m.addVars(Q, lb = 0, vtype = GRB.INTEGER) # Firing for each quarter


U = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Use raw materials
R = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Purchasing raw materials
V = m.addVars(T, raws, lb = 0, vtype=GRB.BINARY)  # Order required 

I_init = m.addVars(goods, lb = 0, ub = 20, vtype = GRB.INTEGER) # Initial inventory
    # Note: We are only allowed up to 20 units in our starting inventory

I_r = m.addVars(T, raws, lb = 0, vtype=GRB.INTEGER) # Inventory of raw materials    
I_r_init = m.addVars(raws, lb = 0, ub = 50, vtype=GRB.INTEGER) # Initial raw inventory
    # Note: we are only allowed up to 50 units in our starting inventory

# Scenario-Dependent Decision Variables
S = m.addVars(T, goods, N, lb = 0, vtype=GRB.INTEGER) # Sales - because of demand
I = m.addVars(T, goods, N, lb = 0, vtype=GRB.INTEGER) # Inventory - because of sales


## Inventory balance constraints

# Initial
m.addConstrs(I_init[g] + P[0, g] - S[0, g, n] == I[0, g, n] for g in goods for n in N)

# No more than 20 total initial units in inventory
m.addConstr(gp.quicksum(I_init[g] for g in goods) <= 20)

# Balance
m.addConstrs(I[t, g, n] == I[t-1, g, n] + P[t-1, g] - S[t, g, n] for g in goods for t in T[1:] for n in N)

# Sales can never be higher than demand
m.addConstrs(S[t, g, n] <= D[t, g, n] for t in T for g in goods for n in N)
    # This allows profits to be tied to the variations in demand


## Labor and overtime constraints

# Labor in each period must a third of the labor in the quarter
m.addConstrs(L[t] == (1/3) * L_quarter[q] for q in Q for t in range(q*3, (q+1)*3))

# Labor in the quarter must be a multiple of three
m.addConstrs(L_quarter[q] == 3 * aux[q] for q in Q)


# Hiring and Firing
m.addConstr(L_quarter[0] == 300 + H[0] - F[0]) # Initial
m.addConstrs(L_quarter[q] == L_quarter[q-1] + H[q-1] - F[q-1] for q in Q[1:]) # Rest of the quarters

# Overtime
m.addConstrs(O[t] <= 0.5 * L[t] for t in T)

# Labor limiting production
m.addConstrs((1-sick[t, n]) * (L[t] + O[t]) >= gp.quicksum(P[t, g] * labor_requirements[g] for g in goods) for t in T for n in N)

## Raw Material Requirements

# Initial
m.addConstrs(I_r_init[r] - U[0, r] == I_r[0, r] for r in [0, 1, 3])

# Balance - All the parts with one week lead time are easy
m.addConstrs(I_r[t,r] == I_r[t-1, r] + expected_quality[r] * R[t-1, r] - U[t, r] for r in [0, 1, 3] for t in T[1:])

# Strawberries - lead time is 2 weeks, which makes life a little more tricky
m.addConstr(I_r[0, 2] == I_r_init[2] - U[0, 2]) # First Time period
m.addConstr(I_r[1, 2] == I_r[0, 2] - U[1, 2]) # Second time period
m.addConstrs(I_r[t, 2] == I_r[t-1, 2] - U[t, 2] + expected_quality[2] * R[t-2, 2] for t in T[2:]) # Rest of the periods

# Raw Materials limiting Production
m.addConstrs(P[t, 0] + P[t, 2] == U[t, 0] for t in T) # Blueberries  used for A and C
m.addConstrs(P[t, 0] + P[t, 1] == U[t, 1] for t in T) #    Pears     used for A and B
m.addConstrs(P[t, 1] == U[t, 2] for t in T)           # Strawberries used for B
m.addConstrs(P[t, 2] == U[t, 3] for t in T)           #   Bananas    used for C

## Set Ups

# First, determine when set-ups are required
m.addConstrs(P[t, g] <= 200 * X[t, g] for g in goods for t in T)
    # Note: 200 is based on observations from previous solutions
    
# m.addConstrs(X[t, g] <= P[t, g] for g in goods for t in T)

# Second, determine if a set-up can be carried over
m.addConstrs(Y[t,g] <= X[t, g] + X[t-1, g] for g in goods for t in T[1:])

# Finally, only save one set up
m.addConstrs(Z[t] <= gp.quicksum(Y[t,g] for g in goods) for t in T[1:])

# Raw material orders
m.addConstrs(R[t, r] <= 200 * V[t, r] for r in raws for t in T)




    
## Objective Function
m.setObjective(gp.quicksum((1/num_scenarios) * c_S[g] * S[t, g, n] for g in goods for t in T for n in N)
             + gp.quicksum(c_X * X[t, g] for g in goods for t in T)
             + gp.quicksum(c_Z * Z[t] for t in T[1:])
             + gp.quicksum((1/num_scenarios) * c_I[g] * I[t, g, n] for g in goods for t in T for n in N)
             + gp.quicksum(c_Ir[r] * I_r[t, r] for r in raws for t in T)
             + gp.quicksum(c_L * L[t] for t in T)
             + gp.quicksum(c_O * O[t] for t in T)
             + gp.quicksum(c_I_init[g] * I_init[g] for g in goods)
             + gp.quicksum(c_Ir_init[r] * I_r_init[r] for r in raws)
             + gp.quicksum(c_V * V[t, r] for t in T for r in raws)
             + gp.quicksum(c_H * H[q] for q in Q)
             + gp.quicksum(c_F * F[q] for q in Q)
             )


m.optimize()

# We use the package Beautiful Table to make our tables, well, beautiful

# Aggregate Production Plan
print("\n\nAggregate Production Plan:")
APP = BeautifulTable()
APP.rows.append(["---"] + [d_means[q*3:(q+1)*3].sum() for q in Q])
APP.rows.append(["---"] + [f"{sum(P[t, g].X for t in range(q*3, (q+1)*3) for g in goods):.0f}" for q in Q])
APP.rows.append(["---"] + [f"{sum(L[t].X + O[t].X for t in range(q*3, (q+1)*3)):.0f}" for q in Q])
APP.rows.append([int(sum(I_init[g].X for g in goods))] + [int(sum(I[q*3+2, g, n].X for g in goods for n in N) / num_scenarios) for q in Q])
APP.rows.header = ["Demand (units)", "Production (units)", "Total Labor Hours", "Inventory"]
APP.columns.header = ["Initial", "Q1", "Q2", "Q3"]
APP.set_style(BeautifulTable.STYLE_GRID)
print(APP)

# Aggregate Production Plan Labor Hour Summary
print("\n\nAggregate Production Plan – Labor Hour Summary")
APL = BeautifulTable()
APL.rows.append([f"{sum(L[t].X for t in range(q*3, (q+1)*3)):.0f}" for q in Q])
APL.rows.append([f"{sum(O[t].X for t in range(q*3, (q+1)*3)):.0f}" for q in Q])
APL.rows.append([f"{sum(L[t].X + O[t].X for t in range(q*3, (q+1)*3)):.0f}" for q in Q])
APL.rows.header = ["Regular Time Labor Hours",
                   "Overtime Labor Hours",
                   "Total Labor Hours"]
APL.columns.header = [f"Q{q+1}" for q in Q]
APL.set_style(BeautifulTable.STYLE_GRID)
print(APL)


# Disaggregate Production Plan
print("\n\nDisaggregate Plan")
DP = BeautifulTable()
for g in goods:
    DP.rows.append([int(I_init[g].X)] + [int(sum(P[t, g].X for t in range(q*3, (q+1)*3))) for q in Q])
DP.rows.append([int(sum(I_init[g].X for g in goods))] + [int(sum(P[t, g].X for t in range(q*3, (q+1)*3) for g in goods)) for q in Q])
DP.rows.header = [f"Product {i}" for i in ["A", "B", "C"]] + ["Total"]
DP.columns.header = ["Initial Inventory"] + [f"Q{q+1}" for q in Q]
DP.set_style(BeautifulTable.STYLE_GRID)
print(DP)

# Master Production Schedule
print("\n\nMaster Production Schedule")
MPS = BeautifulTable(maxwidth=100)
for g in goods:
    MPS.rows.append([int(I_init[g].X)] + [int(P[t, g].X) for t in T])
MPS.rows.append(["--"] + [int(L[t].X + O[t].X) for t in T])
MPS.rows.append(["--"] + [int(L[t].X) for t in T])
MPS.rows.append(["--"] + [int(O[t].X) for t in T])
MPS.rows.header = [f"Product {i}" for i in ["A", "B", "C"]] + ["Labor Hours", "Regular Hours", "Overtime Hours"]
MPS.columns.header = ["Initial"] + [f"T{t+1}" for t in T]
MPS.set_style(BeautifulTable.STYLE_GRID)
print(MPS)
# Be careful, this table is quite wide
    # If you have insufficient characters per line it won't line up
    


# Materials Requirement Plan
names = ["Blueberries", "Pears", "Strawberries", "Bananas"]
# raw_lead_times = [1, 1, 2, 1] # B, G, R, Y
for i in range(len(names)):
    print(f"\n\n{names[i]} Materials requirements")
    MR = BeautifulTable(maxwidth=100)
    MR.rows.append(["---"] + [f"{U[t, i].X:.0f}" for t in T])
    MR.rows.append(["---"] + ["---" for t in T])
    MR.rows.append([f"{I_r_init[i].X:.0f}"] + ["---" for t in T])
    requirements = [f"{U[0,i].X - I_r_init[i].X:.0f}"] + [f"{U[t,i].X:.0f}" for t in T[1:]]
    MR.rows.append(["---"] + requirements)
    MR.rows.append(["---"] + requirements[raw_lead_times[i]:] + ["--"] * raw_lead_times[i])
    MR.rows.append(["---"] + [f"{R[t, i].X:.0f}" for t in T])
    MR.rows.append(["---"] + ["---"]*raw_lead_times[i] + [f"{expected_quality[i] * R[t, i].X:.0f}" for t in T[:-raw_lead_times[i]]])
    MR.rows.append([f"{I_r_init[i].X:.0f}"] + [f"{I_r[t, i].X:.0f}" for t in T])
    MR.columns.header = ["Initial"] + [f"T{t+1}" for t in T]
    MR.rows.header = ["Gross Requirements",
                       "Scheduled Receipts",
                       "Proj. On-hand Inv.",
                       "Net Requirements",
                       "Time Phased Req",
                       "Planned Order Release",
                       "Planned Order Delivery",
                       "Proj. Ending Inv."]
    MR.set_style(BeautifulTable.STYLE_GRID)
    print(MR)




