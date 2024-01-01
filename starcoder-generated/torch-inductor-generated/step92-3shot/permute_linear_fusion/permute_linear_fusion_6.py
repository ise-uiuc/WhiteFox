
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.linear(v3, self.linear3.weight, self.linear3.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 1)
