
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4)
    def forward(self, x):
        v1 = torch.mm(x, self.linear.weight)
        v2 = torch.mm(x, self.linear.weight)
        return torch.cat([v1, v1, v2, v2], 1)
# Inputs to the model
x = torch.randn(2, 10)
