
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224 * 224 * 3, 500)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * torch.clamp(torch.min(v1) + 3, 0, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224 * 224 * 3)
