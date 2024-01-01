
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.tanh(v2)
        v4 = v3 * torch.sigmoid(v2) + torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        y = torch.abs(v4) * torch.tensor(3, dtype=torch.float32) + torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
x5 = torch.randn(1, 2, 2)
x6 = torch.randn(1, 2, 2)
x7 = torch.randn(1, 2, 2)
x8 = torch.randn(1, 2, 2)
x9 = torch.randn(1, 2, 2)
x10 = torch.randn(1, 2, 2)
x11 = torch.randn(1, 2, 2)
x12 = torch.randn(1, 2, 2)
x13 = torch.randn(1, 2, 2)
x14 = torch.randn(1, 2, 2)
x15 = torch.randn(1, 2, 2)
x16 = torch.randn(1, 2, 2)
