import casadi as ca
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# ----- Generate demand needs - URLLC - ARMA(1,1) -----
# t = 40
T = 100
# # URLLC
ar_u = np.array([1, -0.6])     # AR   d_t - 0.6 * d_{t-1}
ma_u = np.array([1, 0.4])      # MA   = e_t + 0.4 * e_{t-1}    e: white noise - N(0,1)
arma_u = ArmaProcess(ar_u, ma_u)
np.random.seed(0)
demands_u = arma_u.generate_sample(T) + 15 #Rouphly [0, 25]
demands_u = np.where(demands_u < 0, 0, demands_u)
demands_u = demands_u.astype(int)

# # eMBB
# ar_e = np.array([1, -0.6])     # AR   d_t - 0.6 * d_{t-1}
# ma_e = np.array([1, 0.4])      # MA   = e_t + 0.4 * e_{t-1}    e: white noise - N(0,1)
# arma_e = ArmaProcess(ar_e, ma_e)
# np.random.seed(1)
# demands_e = arma_e.generate_sample(t) + 15 #Rouphly [0, 25]
# demands_e = np.where(demands_e < 0, 0, demands_e)
# demands_e = demands_e.astype(int)
# # Spike
# for i in range(5):
#     demands_e[i+10] = 100
# ----- Generate demand needs - ARMA(1,1) -----



# ----- Generate demand needs eMBB - MMPP -----
np.random.seed(0)
# State 0 (blue): normal; State 1 (yellow): spike
state_rates = [15,100]  # lambda
P = np.array([[0.8, 0.2], # Transition probability matrix
              [0.9, 0.1]])

time = 0
state = 0
event_times = []
states = []

# def next_state(current, P):
#     rates = P[current, :].copy()
#     rates[current] = 0
#     probs = rates / -P[current, current]
#     return np.random.choice([0, 1], p=probs)

while time < T:
    state_duration = np.random.geometric(p=P[state, abs(state - 1)])  # state change(spike happens) ~ Geo(p)
    t_end = time + state_duration
    if t_end > T:
        t_end = T

    states.append((time, t_end, state))
    
    lam = state_rates[state]
    t_event = time
    while True:
        t_event += np.random.exponential(scale=1 / lam)
        if t_event > t_end or t_event > T:
            break
        event_times.append(t_event)

    time = t_end
    if time >= T:
        break
    state = abs(state - 1)


# plt.subplot(2, 1, 1)
# plt.vlines(event_times, ymin=0, ymax=1, colors='black', label='Arrival', linewidth=0.7)
# ax = plt.gca()
# for start, end, s in states:
#     color = 'orange' if s == 1 else 'deepskyblue'
#     ax.hlines(y=0, xmin=start, xmax=end, colors=color, linewidth=6)
# plt.xlabel('Time')
# plt.title('Arrival Events')
# plt.yticks([])
# plt.legend()
# plt.tight_layout()

bin_edges = np.arange(0, T+1, 1)
demands_e, _ = np.histogram(event_times, bins=bin_edges)

# plt.subplot(2, 1, 2)
# plt.bar(bin_edges[:-1], demands, width=1.0, align='edge', edgecolor='black')
# plt.xlabel("Time")
# plt.ylabel("Number of arrivals")
# plt.title("Traffic")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ----- Generate demand needs - MMPP -----





# 固定参数
r_u = 1
r_e = 1
gamma_u = 10
gamma_e = 10
alpha = 0.3
delta = 0
B_total = 30
threshold = 50
x_u_o=np.zeros(T)
x_e_o=np.zeros(T)
L_o = np.zeros(T)
# grad_L = ca.gradient(L, ca.vertcat(x_u, x_e))

for i in range(T):
    l_max = -1e10
    
    # ----- delta -----
    if demands_e[i] > threshold:
        delta = 1
    else:
        delta = 0
    # ----- delta -----
    
    for u in range(demands_u[i] + 1):
        for e in range(demands_e[i] + 1):
            x_u = u
            x_e = e
            # L = r_u * np.log(x_u + 1) + r_e * np.log(x_e + 1) \
            L = r_u * x_u + r_e * x_e \
                - gamma_u * max(demands_u[i] - x_u, 0) - gamma_e * max(demands_e[i] - x_e, 0)\
                - ((alpha * B_total - x_u) * delta) \
                - ((x_e - min(demands_e[i], B_total - alpha * B_total)) * delta)
                # - (demands_u[i] - x_u)
            if L > l_max and (x_u + x_e - B_total) <= 0 :
                l_max = L
                x_u_o[i] = x_u
                x_e_o[i] = x_e
            # print(x_u, x_e, L)
    print('final:', x_u_o[i], x_e_o[i], l_max)
    L_o[i] = l_max
    
t_x = np.arange(1, T + 1)
# plotting the demands need samples
plt.subplot(3, 1, 1)
plt.plot(t_x, demands_u, label='URLLC Demand', color='blue')
plt.plot(t_x, demands_e, label='eMBB Demand', color='green')
plt.title('Demand (ARMA(1,1))')
plt.xlabel('Time Step')
plt.xlim(0, T + 1)
plt.ylabel('Demand Value')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.bar(t_x, x_e_o, label='eMBB', color='#2AAE50') #green
plt.bar(t_x, x_u_o, bottom=x_e_o, label='URLLC', color='#A9DDFD') #blue
plt.bar(t_x, B_total-x_u_o-x_e_o, bottom=x_u_o + x_e_o, label='Remaining', color='#C9C9C9') #gray
plt.title('PRBs Allocation')
plt.xlabel("Time Step")
plt.xlim(0, T + 1)
plt.ylabel("Allocation")
plt.ylim(0, B_total + 1)
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.plot(t_x, L_o, label='L', color='red')
plt.title('L')

plt.show()

