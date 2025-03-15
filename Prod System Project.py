import tensorflow as tf
import numpy as np
import unittest

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# =============================
# Problem Data and Parameters
# =============================
T = 3  # Three quarters
# Demand per quarter: (for aggregate plan, demand is given per quarter)
demand = [150.0, 300.0, 240.0]

# Cost parameters
material_cost = 15.0          # $15 per unit produced
labor_hours_per_unit = 3.0    # 3 hours needed per unit
regular_rate = 12.0           # $12 per hour
overtime_rate = 18.0          # $18 per hour
regular_hours_available = 300.0  # per quarter
inventory_cost = 15.0         # $15 per unit per quarter

# Increase penalty weight to force no backorders
penalty_weight = 1e12

# ============================================================
# Define production decision variable (raw) and initialization
# ============================================================
# Instead of starting from a low guess, initialize at demand levels so that production equals demand
initial_guess = tf.Variable([200.0, 300.0, 240.0], dtype=tf.float32)
P_raw = tf.Variable(initial_guess, dtype=tf.float32)

# ===============================
# Total Cost Function Definition
# ===============================
def total_cost(P_raw):
    # Use softplus to ensure production is nonnegative
    P = tf.nn.softplus(P_raw)
    
    # Production cost: material cost per unit produced
    prod_cost = material_cost * tf.reduce_sum(P)
    
    # Labor cost: production requires labor_hours_per_unit per unit.
    labor_cost = 0.0
    for t in range(T):
        required_hours = labor_hours_per_unit * P[t]
        reg_hours = tf.minimum(required_hours, regular_hours_available)
        ot_hours = tf.maximum(required_hours - regular_hours_available, 0.0)
        labor_cost += regular_rate * reg_hours + overtime_rate * ot_hours
    
    # Inventory balance, inventory holding cost, and backorder penalty
    I_prev = 0.0  # starting inventory assumed to be 0
    inv_cost = 0.0
    penalty = 0.0
    for t in range(T):
        # Inventory balance: inventory at end of period = previous inventory + production - demand
        I_t = I_prev + P[t] - demand[t]
        # Holding cost applies only on positive inventory
        inv_cost += inventory_cost * tf.maximum(I_t, 0.0)
        # Apply a heavy penalty if inventory is negative (backorders)
        penalty += penalty_weight * tf.maximum(-I_t, 0.0)
        I_prev = I_t

    total = prod_cost + labor_cost + inv_cost + penalty
    return total

# =================================
# Optimization Loop with Adam Optimizer
# =================================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
num_iterations = 1000

for i in range(num_iterations):
    with tf.GradientTape() as tape:
        cost = total_cost(P_raw)
    grads = tape.gradient(cost, [P_raw])
    optimizer.apply_gradients(zip(grads, [P_raw]))
    if i % 100 == 0:
        print(f"Iteration {i}: Total Cost = {cost.numpy():.2f}")

# Obtain final production plan (using softplus)
final_production = tf.nn.softplus(P_raw).numpy()
print("\nOptimal Production Plan (units per quarter):")
for t in range(T):
    print(f"Quarter {t+1}: {final_production[t]:.2f} units")

# Compute final inventory levels for each quarter
I_final = []
I_prev = 0.0
for t in range(T):
    I_t = I_prev + final_production[t] - demand[t]
    I_final.append(I_t)
    I_prev = I_t
print("\nFinal Inventory Levels (units):", I_final)

# =================================
# Unit Tests for the Production Plan
# =================================
class TestProductionPlan(unittest.TestCase):
    def test_no_backorders(self):
        # Each quarter's inventory must be nonnegative.
        for inv in I_final:
            self.assertGreaterEqual(inv, 0, "Negative inventory detected!")

    def test_cost_decrease(self):
        # In a feasible plan, adding extra production should increase total cost.
        perturbed_production = final_production + 10.0  # add 10 units to each quarter
        
        def cost_given_P(P_arr):
            prod_cost = material_cost * np.sum(P_arr)
            labor_cost = 0.0
            I_prev_local = 0.0
            inv_cost = 0.0
            penalty = 0.0
            for t in range(T):
                required_hours = labor_hours_per_unit * P_arr[t]
                reg_hours = min(required_hours, regular_hours_available)
                ot_hours = max(required_hours - regular_hours_available, 0.0)
                labor_cost += regular_rate * reg_hours + overtime_rate * ot_hours
                I_t = I_prev_local + P_arr[t] - demand[t]
                inv_cost += inventory_cost * max(I_t, 0.0)
                penalty += penalty_weight * max(-I_t, 0.0)
                I_prev_local = I_t
            return prod_cost + labor_cost + inv_cost + penalty
        
        cost_original = total_cost(P_raw).numpy()
        cost_perturbed = cost_given_P(perturbed_production)
        self.assertGreater(cost_perturbed, cost_original, "Perturbed production did not increase cost!")

# Run the unit tests
unittest.main(argv=[''], verbosity=2, exit=False)
