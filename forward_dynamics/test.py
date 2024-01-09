import numpy as np
import pickle
import matplotlib.pyplot as plt
with open("/home/lawrence/xembody/robomimic/forward_dynamics/lift_bc_forward_dynamics_data_test.pkl", 'rb') as f:
    a = pickle.load(f)
breakpoint()

predicted_states = np.vstack([a[i]['predicted_state'] for i in range(len(a))])
actual_states = np.vstack([a[i]['next_state'] for i in range(len(a))])
for i in range(9):
    plt.plot(predicted_states[:, i], label=f"predict state {i}")
    plt.plot(actual_states[:, i], label=f"actual state {i}")
    plt.legend()
    plt.savefig(f"/home/lawrence/xembody/robomimic/forward_dynamics/state_{i}.png")
    plt.close()