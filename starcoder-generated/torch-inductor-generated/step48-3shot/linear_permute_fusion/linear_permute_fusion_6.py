
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
    def forward(self, x1):
        v0 = x1
        v1 = torch.nn.functional.linear(v0, self.linear1.weight, self.linear1.bias)
        v4 = v1 + torch.nn.functional.linear(v0, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        v2 = v3.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
