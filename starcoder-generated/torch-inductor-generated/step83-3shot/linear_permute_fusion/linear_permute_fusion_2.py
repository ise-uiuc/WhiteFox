
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(2, 4)
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x0, x1, x2):
        v0 = x0
        v1 = F.linear(v0, self.linear0.weight, self.linear0.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2 + x1
        v4 = F.linear(v3, self.linear1.weight, self.linear1.bias)
        out = v4 + x2
        return out
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 2, 1)
