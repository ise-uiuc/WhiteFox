
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x):
        v1 = torch.nn.functional.hardtanh(x / self.linear1.weight)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        x1 = torch.nn.functional.hardtanh(v1 / v2)
        v3 = torch.ops.aten.max(x1, dim=1, keepdim=True)
        v4 = torch.ops.aten.sinh(v3)
        v5 = torch.nn.functional.softmax(x1 / v4, dim=8)
        v6 = torch.nn.functional.hardtanh(x1 / v5)
        v7 = torch.ops.aten.cos(v6)
        v8 = torch.nn.functional.tanh(x1 / v7)
        v9 = v8 + v8
        v10 = v9 + v7
        v11 = v10 - v5
        v12 = v10.permute(0, 2, 1)
        v13 = torch.ops.aten.max(v11, dim=2, keepdim=True)
        v14 = torch.nn.functional.tanh(v13 * v12)
        v14 = v11.permute(0, 2, 1)
        return torch.nn.functional.sigmoid(v8 * v11)
# Inputs to the model
x = torch.randn(2, 3, 4, 1, 1)
