
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, bias=self.linear1.bias)
        v2 = torch.nn.functional.linear(x2, self.linear2.weight, bias=self.linear2.bias)
        v3 = v1.permute(0, 1, 3, 2)
        v4 = v2.permute(0, 1, 3, 2)
        return torch.cat([v3, v4], dim=1)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 3)
x2 = torch.randn(1, 2, 2, 3)
