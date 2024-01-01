
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 3)
 
    def forward(self, x):
        x = self.fc1(x)
        x = nn.Functional.relu(x)
        x = self.fc2(x)
        x = nn.Functional.relu(x)
        x = nn.Functional.relu(x)
        x = self.fc2(x)
        x = nn.Functional.relu(x)
        x = self.fc2(x)
        x = nn.Functional.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2)
