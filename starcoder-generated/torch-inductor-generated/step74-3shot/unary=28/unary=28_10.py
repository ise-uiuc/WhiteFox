
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 16)
        self.relu = torch.nn.ReLU()

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.relu(v1)
        v3 = torch.relu(v2, 0, 6) # Clamp the output of the previous operation to [0, 6]
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
