
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(12, 8)
        self.linear.weight = torch.nn.Parameter(other)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
x1 = torch.randn(1, 8)
other = torch.zeros(8, 8) + 0.23576802
m = Model(other)

