import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import wandb
import torch.nn as nn
import pypose as pp

# Define a custom dataset to load transitions with images
class TransitionsDataset(Dataset):
    def __init__(self, transitions_files_list):
        self.transitions = []
        for transitions_file in transitions_files_list:
            with open(transitions_file, 'rb') as f:
                self.transitions.extend(pickle.load(f))

        self.current_state = [transition['current_state'] for transition in self.transitions]
        self.actions = [transition['action'] for transition in self.transitions]
        self.next_state = [transition['next_state'] for transition in self.transitions]

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        current_state = torch.tensor(self.current_state[idx], dtype=torch.float) # (9,)
        actions = torch.tensor(self.actions[idx], dtype=torch.float) # (7,)
        next_state = torch.tensor(self.next_state[idx], dtype=torch.float) # (9,)
        inputs = torch.cat((current_state, actions), dim=-1)
        return inputs, next_state
        
# Load transitions data using the custom dataset
transitions_files_list = [
                          '/home/lawrence/xembody/robomimic/forward_dynamics/lift_bc_forward_dynamics_data2.pkl',
                          '/home/lawrence/xembody/robomimic/forward_dynamics/lift_bc_forward_dynamics_data3.pkl'
                          ]
transitions_dataset = TransitionsDataset(transitions_files_list)

# Define batch size and create a DataLoader with prefetching
batch_size = 64
data_loader = DataLoader(transitions_dataset, batch_size=batch_size, shuffle=True)

class PoseBasedForwardDynamicsModel(nn.Module):

    def __init__(self, hidden_dims=[32]):
        super().__init__()
        layers = []
        layers.append(nn.Linear(16, hidden_dims[0]))
        layers.append(nn.ReLU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                # 6 for se3 output, 2 for grippers
                layers.append(nn.Linear(hidden_dims[l], 8))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(nn.ReLU())
        self.delta_transform_output = nn.Sequential(*layers)

    def forward(self, state_action):
        state = state_action[..., :9]
        action = state_action[..., 9:]
        
        se3_gripper_act = self.delta_transform_output(state_action)
        init_pose = pp.SE3(state[..., :7])
        computed_end_pose = pp.se3(se3_gripper_act[..., :6]).Exp() @ init_pose
        return computed_end_pose, se3_gripper_act[..., 6:]


pose_hidden_dims = [32, 128, 128]
forward_dynamics_model = PoseBasedForwardDynamicsModel(hidden_dims=pose_hidden_dims)
forward_dynamics_model.to("cuda")

criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(forward_dynamics_model.parameters(), lr=learning_rate)

num_epochs = 800

gripper_diff_weight = 1
ee_diff_weight = 8


wandb.init(
    entity="xembody",
    project="forward-dynamics-model",
    config={
        "transition_files": transitions_files_list,
        "lr": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "hidden_dims": pose_hidden_dims,
        "gripper_diff_weight": gripper_diff_weight,
        "ee_diff_weight": ee_diff_weight
    }
)

# Training loop
for epoch in range(num_epochs):
    running_pose_diff_loss = 0.0
    running_gripper_diff_loss = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        targets_end_pose = pp.SE3(targets[..., :7])
        targets_gripper_actions = targets[..., 7:]

        optimizer.zero_grad()
        computed_end_pose, gripper_actions = forward_dynamics_model(inputs)
        pose_difference_loss = ee_diff_weight * (computed_end_pose.Inv() @ targets_end_pose).Log().abs().mean()
        gripper_difference_loss = gripper_diff_weight * criterion(gripper_actions, targets_gripper_actions)
        
        loss = gripper_difference_loss + pose_difference_loss
        
        loss.backward()
        optimizer.step()
        running_pose_diff_loss += pose_difference_loss.item()
        running_gripper_diff_loss += gripper_difference_loss.item()
        
    # Log loss to wandb
    wandb.log({
        "train/running_pose_diff_loss": running_pose_diff_loss,
        "train/running_gripper_diff_loss": running_gripper_diff_loss
    })

    # save checkpoint
    if (epoch + 1) % 50 == 0:
        torch.save(forward_dynamics_model.state_dict(), f'/home/lawrence/xembody/robomimic/forward_dynamics_1layer_32_physics/forward_dynamics_bc_img_{epoch + 1}.pth')
        print('Saved checkpoint at step', epoch)

"""
# inspect model prediction
import matplotlib.pyplot as plt
import numpy as np

# Load transitions data
with open('/home/lawrence/xembody/robomimic/forward_dynamics/lift_bc_forward_dynamics_data3.pkl', 'rb') as f:
    transitions = pickle.load(f)

# load policy checkpoint
forward_dynamics_model.load_state_dict(torch.load('/home/lawrence/xembody/robomimic/forward_dynamics_1layer/forward_dynamics_bc_img_50.pth'))

actions = []
predicted_states = []
actual_states = []
for i in range(2000, min(4000, len(transitions)), 1):
    transition = transitions[i]
    current_state = transition['current_state']
    action = transition['action']
    next_state = transition['next_state']
    inputs = torch.cat((torch.tensor(current_state, dtype=torch.float), torch.tensor(action, dtype=torch.float)), dim=-1).unsqueeze(0).to("cuda")
    
    predicted_state = forward_dynamics_model(inputs)
    predicted_state = predicted_state.cpu().detach().numpy()
    
    predicted_states.append(predicted_state.copy())
    actual_states.append(next_state.copy())
    actions.append(action.copy())

predicted_states = np.vstack(predicted_states)
actual_states = np.vstack(actual_states)
actions = np.vstack(actions)
for i in range(9):
    plt.plot(predicted_states[:, i], label=f"predicted state {i}")
    plt.plot(actual_states[:, i], label=f"actual state {i}")
    plt.legend()
    plt.savefig(f"/home/lawrence/xembody/robomimic/forward_dynamics_1layer/state_{i}.png")
    plt.close()
    
# plot the 7D action horizontally in a plot consisting of 7 subplots
for i in range(7):
    plt.subplot(7, 1, i + 1)
    plt.plot(actions[:, i], label=f"action {i}")
    plt.legend()
plt.savefig(f"/home/lawrence/xembody/robomimic/forward_dynamics_1layer/action.png")
plt.close()
"""