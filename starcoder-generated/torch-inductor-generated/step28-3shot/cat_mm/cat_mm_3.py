
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v1 = torch.mm(x, self.linear.weight)
        v2 = torch.mm(x, self.linear.weight)
        v3 = torch.mm(x, self.linear.weight)
        return torch.cat([v1, v2, v3], 1)
# Inputs to the model
x = torch.randn(1, 2)
