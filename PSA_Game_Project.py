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

num_scenarios = 50000 # for now
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
# In a specific scenario, only one of the periods should have absentees
    # Specifically, one of the last three periods
# Up to 50% absent

# First, determine which of the three periods will have the pandemic
pandemic = np.random.randint(6, 9, num_scenarios)

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

# Balance
m.addConstrs(I[t, g, n] == I[t-1, g, n] + P[t-1, g] - S[t, g, n] for g in goods for t in T[1:] for n in N)

# Sales can never be higher than demand
m.addConstrs(S[t, g, n] <= D[t, g, n] for t in T for g in goods for n in N)
m.addConstrs(S[t, g, n] <= I[t, g, n] for t in T for g in goods for n in N) # Technically unecessary

    # This allows profits to be tied to the variations in demand


## Labor and overtime constraints

# Labor in each period must a third of the labor in the quarter
m.addConstrs(L[t] == (1/3) * L_quarter[q] for q in Q for t in range((q+1)*3-3, (q+1)*3))

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
m.addConstrs(X[t, g] <= P[t, g] for g in goods for t in T)

# Second, determine if a set-up can be carried over
m.addConstrs(Y[t,g] <= X[t, g] + X[t-1, g] for g in goods for t in T[1:])

# Finally, only save one set up
m.addConstrs(Z[t] <= gp.quicksum(Y[t,g] for g in goods) for t in T[1:])

# Raw material orders
m.addConstrs(R[t, r] <= 1000 * V[t, r] for r in raws for t in T)




    
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
# TDL:
    # Labor Shortage - Diran or Ben
    # Objective function - Ben

# Aggregate Production Plan
print("\nAggregate Production Plan")
print("{:<8} {:<12} {:<20} {:<20} {:<15}".format("Quarter", "Demand", "Prod. Units", "Total Labor Hours", "Inventory"))
for q in Q:
    # Calculate the total demand for the quarter using the mean demands
    quarter_demand = d_means[q*3:(q+1)*3].sum()
    # Sum production over all periods in the quarter across all products
    quarter_production = sum(P[t, g].X for t in range(q*3, (q+1)*3) for g in goods)
    # Total labor hours is the sum of regular and overtime labor for the quarter
    quarter_labor = sum(L[t].X + O[t].X for t in range(q*3, (q+1)*3))
    # Average inventory at the end of the quarter (using the inventory at period q*3+2)
    quarter_inventory = sum(I[q*3+2, g, n].X for g in goods for n in N) / num_scenarios
    print("{:<8} {:<12} {:<20} {:<20} {:<15.2f}".format(q+1, quarter_demand, quarter_production, quarter_labor, quarter_inventory))

# Disaggregate Production Plan
print("\nDisaggregate Production Plan")
print("{:<8} {:<10} {:<10} {:<10}".format("Quarter", "Product A", "Product B", "Product C"))
for q in Q:
    prod_A = sum(P[t, 0].X for t in range(q*3, (q+1)*3))
    prod_B = sum(P[t, 1].X for t in range(q*3, (q+1)*3))
    prod_C = sum(P[t, 2].X for t in range(q*3, (q+1)*3))
    print("{:<8} {:<10} {:<10} {:<10}".format(q+1, prod_A, prod_B, prod_C))

# Master Production Schedule
print("\nMaster Production Schedule")
print("{:<8} {:<10} {:<10} {:<10}".format("Period", "Product A", "Product B", "Product C"))
for t in T:
    print("{:<8} {:<10} {:<10} {:<10}".format(t+1, P[t, 0].X, P[t, 1].X, P[t, 2].X))

# Materials Requirement Plan
print("\nMaterials Requirement Plan")
print("{:<8} {:<12} {:<8} {:<12} {:<10}".format("Period", "Blueberry", "Pear", "Strawberry", "Banana"))
for t in T:
    print("{:<8} {:<12} {:<8} {:<12} {:<10}".format(t+1, U[t, 0].X, U[t, 1].X, U[t, 2].X, U[t, 3].X))
