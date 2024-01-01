
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 2)
        self.linear2 = torch.nn.Linear(3, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(x, self.linear2.weight, self.linear2.bias)
        return torch.cat([v1, v2])
# Inputs to the model
x = torch.randn(1, 6)
