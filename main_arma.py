from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

t = 200

# Generate demand needs - ARMA(1,1)
# URLLC
ar_u = np.array([1, -0.6])     # AR   d_t - 0.6 * d_{t-1}
ma_u = np.array([1, 0.4])      # MA   = e_t + 0.4 * e_{t-1}    e: white noise - N(0,1)
arma_u = ArmaProcess(ar_u, ma_u)
np.random.seed(0)
demands_u = arma_u.generate_sample(t) * 5 #Rouphly [0, 25]
demands_u = np.where(demands_u < 0, 0, demands_u)

# eMBB
ar_e = np.array([1, -0.6])     # AR   d_t - 0.6 * d_{t-1}
ma_e = np.array([1, 0.4])      # MA   = e_t + 0.4 * e_{t-1}    e: white noise - N(0,1)
arma_e = ArmaProcess(ar_e, ma_e)
np.random.seed(1)
demands_e = arma_e.generate_sample(t) * 5 #Rouphly [0, 25]
demands_e = np.where(demands_e < 0, 0, demands_e)

# plotting the demands need samples
plt.subplot(2, 1, 1)
plt.plot(demands_u, label='URLLC Demand', color='blue')
plt.title('URLLC Demand (ARMA(1,1))')
plt.ylabel('Demand Value')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(demands_e, label='eMBB Demand', color='green')
plt.title('eMBB Demand (ARMA(1,1))')
plt.xlabel('Time Step')
plt.ylabel('Demand Value')
plt.grid(True)
plt.tight_layout()
plt.show()


r_u = 2 # Reward utility per unit performance for URLLC
r_e = 1
gamma = 5 # penalty wieght for URLLC SLA violation
alpha = 0.3 # resilience budge factor
B_total = 100 # total PRBs
threshold = 90 # threshold for URLLC traffic spike
L = 0
t = 200

x_u = ca.MX.sym('x_u')
x_e = ca.MX.sym('x_e')

for i in range(t):
    if demands_u[i] > threshold:
        delta = 1
    else:
        delta = 0
    L += - (r_u * np.log(x_u + 1) + r_e * np.log(x_e + 1) - gamma * ca.fmax(demands_u - B_total, 0) \
    -(x_u + x_e - B_total) - (alpha * B_total * delta - x_u) - \
        (x_e - ca.fmin(demands_e, B_total - alpha * B_total * delta)) - (demands_u - x_u))
nlp = {
    'x': ca.vertcat(x_u, x_e),
    'f': L
}

# 设置求解器
solver = ca.nlpsol('solver', 'ipopt', nlp)

# 初始猜测
x0 = [1.0, 1.0]

# 可选约束（非负分配）
lbx = [0.0, 0.0]
ubx = [ca.inf, ca.inf]

# 求解
sol = solver(x0=x0, lbx=lbx, ubx=ubx)
x_opt = sol['x']

# 输出结果  
print("Optimal x_u:", float(x_opt[0]))
print("Optimal x_e:", float(x_opt[1]))