
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 * torch.clamp_min(torch.clamp_max(3 + v1, 6), 0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(4, 16)
