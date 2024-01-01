
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32, bias=False)
        self.fc2 = torch.nn.Linear(32, 64)
 
    def forward(self, x, other):
        v0 = torch.relu((self.fc1(x) + other))
        return torch.relu((self.fc2(v0) + other))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 1)
