
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 48)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = torch.min(v2, 0)
        v4 = torch.max(v3, 0)
        v5 = v4[0] / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
