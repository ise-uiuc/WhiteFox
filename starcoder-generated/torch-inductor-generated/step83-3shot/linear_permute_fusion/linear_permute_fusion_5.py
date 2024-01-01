
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 4)
    def forward(self, x0):
        v0 = x0
        v1 = F.linear(v0, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = F.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x0 = torch.randn(1, 2, 2)
