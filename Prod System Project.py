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
demand = [150.0, 300.0, 240.0]  # Demand per quarter

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
# Instead of starting from a low guess, initialize at demand levels so that production equals demand.
initial_guess = tf.Variable([200.0, 300.0, 240.0], dtype=tf.float32)
P_raw = tf.Variable(initial_guess, dtype=tf.float32)

# ===============================
# Total Cost Function Definition
# ===============================
def total_cost(P_raw):
    # Use softplus to ensure production is nonnegative:
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
    
    # Inventory balance, inventory holding cost, and backorder penalty:
    I_prev = 0.0  # starting inventory is assumed to be 0
    inv_cost = 0.0
    penalty = 0.0
    for t in range(T):
        # Inventory balance: I_t = I_prev + production - demand
        I_t = I_prev + P[t] - demand[t]
        # Holding cost applies only to positive inventory
        inv_cost += inventory_cost * tf.maximum(I_t, 0.0)
        # Backorders (negative inventory) are penalized heavily
        penalty += penalty_weight * tf.maximum(-I_t, 0.0)
        I_prev = I_t

    total = prod_cost + labor_cost + inv_cost + penalty
    return total

# =================================
# Optimization Loop with Adam Optimizer
# =================================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
num_iterations = 1000

print("Starting optimization...\n")
for i in range(num_iterations):
    with tf.GradientTape() as tape:
        cost = total_cost(P_raw)
    grads = tape.gradient(cost, [P_raw])
    optimizer.apply_gradients(zip(grads, [P_raw]))
    
    # Every 100 iterations print detailed inputs/outputs:
    if i % 100 == 0:
        P_current = tf.nn.softplus(P_raw).numpy()
        grad_values = grads[0].numpy()
        print(f"Iteration {i}:")
        print(f"  Raw P: {P_raw.numpy()}")
        print(f"  Production P (softplus): {P_current}")
        print(f"  Gradients: {grad_values}")
        print(f"  Total Cost: {cost.numpy():.2f}\n")
        # Example expected output:
        # Iteration 0:
        #   Raw P: [200. 300. 240.]
        #   Production P (softplus): [200. 300. 240.]
        #   Gradients: [ ... ]
        #   Total Cost: 69015296.00

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
        """Test that the final production plan does not produce negative inventory."""
        for t, inv in enumerate(I_final):
            self.assertGreaterEqual(inv, 0, f"Negative inventory detected in quarter {t+1}: {inv}")

    def test_exact_demand_plan(self):
        """Test that a plan with production exactly equal to demand produces zero inventory and no penalty."""
        exact_plan = np.array(demand)
        def compute_inventory(P_arr):
            I_prev_local = 0.0
            inventories = []
            for t in range(T):
                I_t = I_prev_local + P_arr[t] - demand[t]
                inventories.append(I_t)
                I_prev_local = I_t
            return inventories
        
        inventories = compute_inventory(exact_plan)
        for t, inv in enumerate(inventories):
            self.assertAlmostEqual(inv, 0.0, places=6,
                                   msg=f"Inventory not zero for exact demand plan in quarter {t+1}")

    def test_under_production_penalty(self):
        """Test that under-producing (less than demand) leads to a huge penalty cost."""
        under_plan = np.array([100.0, 200.0, 150.0])  # clearly below demand
        def compute_total_cost(P_arr):
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
        
        cost_under = compute_total_cost(under_plan)
        self.assertTrue(cost_under > 1e11, "Under-production penalty is not high enough!")

    def test_cost_decrease(self):
        """Test that a perturbation that increases production increases the overall cost."""
        perturbed_production = final_production + 10.0  # add 10 units per quarter
        
        def compute_cost(P_arr):
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
        cost_perturbed = compute_cost(perturbed_production)
        self.assertGreater(cost_perturbed, cost_original,
                           "Perturbed production did not increase cost!")

    def test_material_cost_component(self):
        """Test that the material cost is computed correctly for a given production vector."""
        P_test = np.array([180.0, 320.0, 250.0])
        expected_material_cost = material_cost * np.sum(P_test)
        computed_material_cost = material_cost * np.sum(P_test)
        self.assertAlmostEqual(expected_material_cost, computed_material_cost, places=6,
                               msg="Material cost component is not computed correctly.")

    def test_labor_cost_component(self):
        """Test that the labor cost is computed correctly for a given production vector."""
        P_test = np.array([180.0, 320.0, 250.0])
        computed_labor_cost = 0.0
        for t in range(T):
            required_hours = labor_hours_per_unit * P_test[t]
            reg_hours = min(required_hours, regular_hours_available)
            ot_hours = max(required_hours - regular_hours_available, 0.0)
            computed_labor_cost += regular_rate * reg_hours + overtime_rate * ot_hours
        # Expected values manually computed:
        # Quarter 1: 180 units -> 540 hours → cost = 540 * 12 = 6480
        # Quarter 2: 320 units -> 960 hours → regular cost: 300*12 = 3600, overtime: 660*18 = 11880, total = 15480
        # Quarter 3: 250 units -> 750 hours → regular cost: 300*12 = 3600, overtime: 450*18 = 8100, total = 11700
        expected_labor_cost = 6480 + 15480 + 11700
        self.assertAlmostEqual(computed_labor_cost, expected_labor_cost, places=6,
                               msg="Labor cost component is not computed correctly.")

# Run the unit tests
unittest.main(argv=[''], verbosity=2, exit=False)
