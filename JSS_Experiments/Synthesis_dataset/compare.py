import pickle
import torch.nn as nn


class Net1(nn.Module):
    def __init__(self, out_put):
        super(Net1, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, out_put)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.output_layer(x)
        return x


with open("JSS_Experiments/Synthesis_dataset/noisy_circles2.pth", "rb") as f:
    model_a = pickle.load(f)


with open("JSS_Experiments/Synthesis_dataset/noisy_circles.pth", "rb") as f2:
    model_b = pickle.load(f2)

# Compare the parameter data
are_models_equal = model_a == model_b

if are_models_equal:
    print("The models have the same parameters.")
else:
    print("The models have different parameters.")
