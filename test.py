import numpy as np
import sys
import os
import do_mpc
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import matplotlib as mpl

rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
model_type = 'discrete' # discrete / continuous
model = do_mpc.model.Model(model_type)

# ----- Create the model -----
demands_e = model.set_variable(var_type='_x', var_name='demands_e', shape=(1,1))
demands_u = model.set_variable(var_type='_x', var_name='demands_u', shape=(1,1))
x_u = model.set_variable(var_type='_x', var_name='x_u', shape=(1,1)) # #PRBs allocated for URLLC
x_e = model.set_variable(var_type='_x', var_name='x_e', shape=(1,1))

x_u_set = model.set_variable(var_type='_u', var_name='x_u_set', shape=(1,1)) # PRBs allocated for URLLC
x_e_set = model.set_variable(var_type='_u', var_name='x_e_set', shape=(1,1)) # PRBs allocated for eMBB

# delta = model.set_variable(var_type='_u', var_name='delta', shape=(1,1)) # whether traffic spike occurs
delta = 0

e = model.set_variable(var_type='_p', var_name='e', shape=(1,1)) # noise
state_e = model.set_variable(var_type='_p', var_name='state_e', shape=(1,1)) # state of eMBB

r_u = 2 # Reward utility per unit performance for URLLC
r_e = 1
alpha = 0.3 # resilience budge factor
B_total = 40 # total PRBs
# threshold = 90 # threshold for URLLC traffic spike
spike = 100
gamma_u = 10 # penalty wieght for URLLC SLA violation
gamma_e = 10
T = 100
tstep = 1

np.random.seed(0)
# State 0 (blue): normal; State 1 (yellow): spike
state_rates = [15,100]  # lambda
P = np.array([[0.9, 0.1], # Transition probability matrix
              [0.9, 0.1]])

time = 0
state = 0
event_times = []
states = []

# Set _x rhs: x{t+1} = f(xt, ut)
model.set_rhs('demands_u', 0.6 * demands_u + e)
# rhs = ca.if_else(state_e == 0, P[0][0]*state_rates[0]+P[0][1]*state_rates[1], P[1][0]*state_rates[0]+P[1][1]*state_rates[1])
rhs = ca.if_else(state_e == 0, 0.6 * demands_e + e, 0.6 * demands_e + spike)
model.set_rhs('demands_e', rhs)
  

tau = 1e-2 # smooth input tracking 平滑追踪
model.set_rhs('x_u', 1/tau*(x_u_set - x_u))
model.set_rhs('x_e', 1/tau*(x_e_set- x_e))
model.setup()

# ----- Configuring the MPC controller -----
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,
    't_step': tstep,
    'n_robust': 1, #controls how many time steps the optimized input is held constant ("blocked"). 如果为2，则u1=u0, u3=u2...
    'store_full_solution': True, # 是否保存完整优化解轨迹
}
mpc.set_param(**setup_mpc)

# ----- Optimizer parameters -----
# Objective function
lterm = -  (r_u * x_u + r_e * x_e \
            - gamma_u * ca.fmax(demands_u - x_u, 0) - gamma_e * ca.fmax(demands_e - x_e, 0)\
            - (x_e + x_u - B_total) * 10000\
            - ((alpha * B_total - x_u) * delta) \
            - ((x_e - ca.fmin(demands_e, B_total - alpha * B_total)) * delta))
mterm = lterm

mpc.set_objective(lterm=lterm, mterm=mterm)
mpc.set_rterm(
    x_e_set = 1e-2,
    x_u_set = 1e-2
)

# Constraints
mpc.bounds['lower','_x', 'demands_u'] = 0
mpc.bounds['upper','_x', 'demands_u'] = B_total
mpc.bounds['lower','_x', 'demands_e'] = 0
mpc.bounds['upper','_x', 'demands_e'] = B_total

mpc.bounds['lower','_u', 'x_u_set'] = 0
mpc.bounds['upper','_u', 'x_u_set'] = B_total
mpc.bounds['lower','_u', 'x_e_set'] = 0
mpc.bounds['upper','_u', 'x_e_set'] = B_total

# Uncertain Parameters
inter_e = np.array([1, 2, 3])

mpc.set_uncertainty_values(
    e = inter_e
)
mpc.setup()

# ----- Configuring the simulator -----
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = tstep)

p_template = simulator.get_p_template()
def p_fun(t_now):
    p_template['e'] = 0.4 * p_template['e'] + abs(np.random.normal(10, 1))
    if p_template['state_e'] == 0:
        p_template['state_e'] = np.random.choice([0, 1], p=P[0])
    else:
        p_template['state_e'] = np.random.choice([0, 1], p=P[1])
    return p_template
simulator.set_p_fun(p_fun)
simulator.setup()

# Creating the control loop
x0 = np.array([20,20,20,20]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()


# ----- Setting up the graphic -----
mpl.rcParams['font.size'] = 15
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()
for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='demands_u', axis=ax[0])
    g.add_line(var_type='_x', var_name='demands_e', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_vm_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='x_u_set', axis=ax[1])
    g.add_line(var_type='_u', var_name='x_e_set', axis=ax[1])

ax[0].set_ylabel('Demands Needs')
ax[0].set_xlabel('time [s]')
ax[0].legend(['demands_u', 'demands_e'], loc='lower right', fontsize=12)
ax[1].set_ylabel('Allocated PRBs')
ax[1].set_xlabel('time [s]')
ax[1].legend(['x_u_set', 'x_e_set'], loc='lower right', fontsize=12)

# ----- Running the simulator - original -----
# u0 = np.zeros((2,1))
# for i in range(T):
#     simulator.make_step(u0) # calculate the next state x{k+1}
# sim_graphics.plot_results()
# # Reset the limits on all axes in graphic to show the data.
# sim_graphics.reset_axes() 

# # ----- Running the optimizer -----
u0 = np.zeros((2,1))
for i in range(T):
    simulator.make_step(u0) # calculate the next state x{k+1}
    u0 = mpc.make_step(x0) # return the first control input u0

sim_graphics.clear()
sim_graphics.plot_results()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()

plt.show()