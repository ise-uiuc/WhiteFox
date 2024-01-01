
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(x1 - v1, self.linear2.weight, self.linear2.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v4, self.linear1.weight, self.linear1.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
