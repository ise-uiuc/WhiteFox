
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(2, 3)
        self.linear3 = torch.nn.Linear(3, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v1 = v1.permute(2, 0, 1)
        v1 = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        v3 = self.linear3(v2)
        v3 = v3.expand(2, 3, 4)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 1)
