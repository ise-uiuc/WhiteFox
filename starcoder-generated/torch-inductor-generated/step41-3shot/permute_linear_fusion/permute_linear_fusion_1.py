
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        x2 = torch.sum(x1, dim=1, keepdim=True) * v3 ** 2
        v4 = torch.max(x1, dim=-1, keepdim=True)[0]
        v5 = v4 ** 2
        x3 = x2 ** 2 + v5
        v6 = x2 ** 5 + torch.max(x2, dim=-1, keepdim=True)[0]
        x2 = v2 * x2
        x4 = torch.sqrt(x2 * v6)
        v7 = v5 * torch.sqrt(x4 * v6) ** 2
        v8 = torch.nn.functional.softmax(v7, dim=-1)
        v8 = v8.permute(0, 2, 1)
        v9 = torch.max(x1, dim=-1, keepdim=True)[0]
        x5 = torch.sum(v8) * torch.sum(v9)
        return x5 + v7.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
