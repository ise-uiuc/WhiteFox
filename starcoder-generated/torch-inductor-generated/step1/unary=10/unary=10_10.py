
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 10)
        self.fc2 = torch.nn.Linear(10, 20)
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.fc2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0., max=6.)
        v5 = v4 / 6.
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
