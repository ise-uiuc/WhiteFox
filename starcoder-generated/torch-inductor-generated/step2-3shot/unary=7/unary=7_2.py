
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 * torch.clamp(torch.nn.functional.relu(v1 + 3, inplace=False), min=0, max=6)
        v3 = v2 / 6.0
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
