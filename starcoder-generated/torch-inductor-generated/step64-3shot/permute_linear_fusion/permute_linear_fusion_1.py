
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.normal = torch.distributions.normal.Normal(0, 2)
    def forward(self, x):
        g1 = x.permute(0, 2, 1)
        g2 = torch.nn.functional.linear(g1, self.linear1.weight, self.linear1.bias)
        g3 = torch.nn.functional.linear(g2 + 2, self.linear2.weight, self.linear2.bias)
        g3 = g3.permute(0, 2, 1)
        return g3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
