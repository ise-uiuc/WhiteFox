
class Model(torch.nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        self.linear = nn.Linear(m1, m2)

    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 * torch.clamp(l1+3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model. The value of `m1` and `m2` are not specified yet.
m = Model(20, 30)

# Inputs to the model
x = torch.randn(1, 20)
