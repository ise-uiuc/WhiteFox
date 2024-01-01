
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(1, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v3 = x1 - v1
        v2 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1)
