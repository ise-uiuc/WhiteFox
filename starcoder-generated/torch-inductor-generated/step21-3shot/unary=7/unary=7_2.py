
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=True)

    def forward(self, x1):
        v1 = self.linear(x1)
        l2 = self.linear.weight * torch.clamp(min=0, max=6, self.linear.bias + 3)
        v3 = l2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
