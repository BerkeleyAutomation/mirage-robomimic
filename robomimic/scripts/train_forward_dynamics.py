import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import multiprocessing
import torch.nn as nn

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



# hidden_dims=[256, 256, 256]
# TODO: play around with this for forward dynamics
hidden_dims=[32]
layers = []
layers.append(nn.Linear(16, hidden_dims[0]))
layers.append(nn.ReLU())
for l in range(len(hidden_dims)):
    if l == len(hidden_dims) - 1:
        layers.append(nn.Linear(hidden_dims[l], 9))
    else:
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(nn.ReLU())
forward_dynamics_model = nn.Sequential(*layers)
forward_dynamics_model.to("cuda")




# """
        
# self.actor = nn.Sequential(self.actor_encoder, self.actor_decoder)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(forward_dynamics_model.parameters(), lr=0.0001)

# set up tensorboard
from torch.utils.tensorboard import SummaryWriter
# Iterate through the DataLoader to train your model
writer = SummaryWriter()

# Training loop
num_epochs = 800
for epoch in range(num_epochs):
    running_loss = 0.0
    # for i, (images, actions0, actions1, actions2, actions3, actions4) in enumerate(data_loader):
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")

        optimizer.zero_grad()
        outputs = forward_dynamics_model(inputs)
        
        loss = criterion(outputs, targets) * 10000
        
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + i)

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            # print(outputs[..., -2], actions[..., -2])
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i + 1}/{len(data_loader)}] "
                    f"Loss: {running_loss / 99:.4f}")
        # running_loss = 0.0
    # save checkpoint
    if (epoch + 1) % 50 == 0:
        print('Step:', epoch, 'Loss:', running_loss)
        torch.save(forward_dynamics_model.state_dict(), f'/home/lawrence/xembody/robomimic/forward_dynamics_1layer_32/forward_dynamics_bc_img_{epoch + 1}.pth')
        print('Saved checkpoint at step', epoch)
# Close the TensorBoard writer
writer.close()
# """

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