
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(16, 8)
        self.fc1 = torch.nn.Linear(8, 4)
 
    def forward(self, x1):
        v1 = self.fc0(x1)
        v2 = v1 + 1
        v3 = self.fc1(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
