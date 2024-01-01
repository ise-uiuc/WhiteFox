
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(64, 20)
        self.fc1 = torch.nn.Linear(20, 40)
        self.fc2 = torch.nn.Linear(40, 64)
 
    def forward(self, x1):
        v1 = torch.tanh(self.fc0(x1))
        v2 = torch.tanh(self.fc1(v1))
        v3 = torch.tanh(self.fc2(v2))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
