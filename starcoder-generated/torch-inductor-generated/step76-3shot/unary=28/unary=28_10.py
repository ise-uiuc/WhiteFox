
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        a1 = self.linear(x)
        a2 = torch.clamp_min(a1, -5)
        a3 = torch.clamp_max(a2, 5)
        return a3 + 1

# Initializing the model
m = Model(-2, 4)

x = torch.randn(1, 4, 1, 1)
