import torch

# Replace this with the path to your model's checkpoint
checkpoint_path = "/home/davery/pytorch_model.bin"

# Load the saved state dictionary
state_dict = torch.load(checkpoint_path)

# Inspect the state dictionary, especially the final layers
for key in state_dict.keys():
    print(key, state_dict[key].shape)
