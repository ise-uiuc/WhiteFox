
class LeakyRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr0 = 0.2

    def forward(self, x):
        v1 = torch.nn.functional.linear(x, torch.ones(1, 96, requires_grad=False))
        v2 = v1 > 0
        v3 = v1 * self.lr0
        return torch.where(v2, v1, v3)

# Initializing the model
m = LeakyRelu()

# Inputs to the model
x = torch.randn(1, 96, requires_grad=True)
