import numpy as np
import matplotlib.pyplot as plt

T = 100
np.random.seed(0)
# State 0 (yellow): normal; State 1 (blue): spike
state_rates = [1, 10]
Q = np.array([[-5, 5],
              [3, -3]])  # Q[i, i] = -∑Q[i, j] total rate out of state i

t = 0.0
state = 0
event_times = []
state_trace = []

def next_state(current, Q):
    rates = Q[current, :].copy()
    rates[current] = 0  # P(self-loop) = 0
    probs = rates / -Q[current, current]
    return np.random.choice([0, 1], p=probs)

while t < T:
    # time the system stays in state i
    state_duration = np.random.exponential(scale=1 / -Q[state, state])
    t_end = t + state_duration
    
    # in current state, generate poisson arrivals
    lam = state_rates[state]
    t_event = t
    while True:
        t_event += np.random.exponential(scale=1 / lam)
        if t_event > t_end or t_event > T:
            break
        event_times.append(t_event)
        state_trace.append(state)

    # calculate the next state
    if t_end > T:
        break
    state = next_state(state, Q)
    t = t_end

plt.subplot(2, 1, 1)
current_time = 0
ax = plt.gca()
for state, event_time in zip(state_trace, event_times):
    color = 'orange' if state == 1 else 'deepskyblue'
    ax.hlines(y=0, xmin=current_time, xmax=event_time, colors=color, linewidth=6)
    current_time = event_time
plt.vlines(event_times, ymin=0, ymax=1, colors='black', label='Arrival', linewidth=0.7)
# plt.plot(event_times)
plt.xlabel('Time')
plt.title('MMPP Arrival Events')
plt.yticks([]) 
plt.legend()
plt.grid(True)
plt.tight_layout()


bin_edges = np.arange(0, T+1, 1)  # 每秒一个区间
counts, _ = np.histogram(event_times, bins=bin_edges)

plt.subplot(2, 1, 2)
plt.bar(bin_edges[:-1], counts, width=1.0, align='edge', edgecolor='black')
plt.xlabel("Time (seconds)")
plt.ylabel("Number of arrivals")
plt.title("Poisson arrivals per second (λ = 5)")
plt.grid(True)
plt.tight_layout()
plt.show()