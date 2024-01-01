

class Model(nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = nn.Linear(3, 8)
        self.negative_slope = negative_slope

    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 3)
